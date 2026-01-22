import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
class Flowchart(Node):
    sigFileLoaded = QtCore.Signal(object)
    sigFileSaved = QtCore.Signal(object)
    sigChartLoaded = QtCore.Signal()
    sigStateChanged = QtCore.Signal()
    sigChartChanged = QtCore.Signal(object, object, object)

    def __init__(self, terminals=None, name=None, filePath=None, library=None):
        self.library = library or LIBRARY
        if name is None:
            name = 'Flowchart'
        if terminals is None:
            terminals = {}
        self.filePath = filePath
        Node.__init__(self, name, allowAddInput=True, allowAddOutput=True)
        self.inputWasSet = False
        self._nodes = {}
        self.nextZVal = 10
        self._widget = None
        self._scene = None
        self.processing = False
        self.widget()
        self.inputNode = Node('Input', allowRemove=False, allowAddOutput=True)
        self.outputNode = Node('Output', allowRemove=False, allowAddInput=True)
        self.addNode(self.inputNode, 'Input', [-150, 0])
        self.addNode(self.outputNode, 'Output', [300, 0])
        self.outputNode.sigOutputChanged.connect(self.outputChanged)
        self.outputNode.sigTerminalRenamed.connect(self.internalTerminalRenamed)
        self.inputNode.sigTerminalRenamed.connect(self.internalTerminalRenamed)
        self.outputNode.sigTerminalRemoved.connect(self.internalTerminalRemoved)
        self.inputNode.sigTerminalRemoved.connect(self.internalTerminalRemoved)
        self.outputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        self.inputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        self.viewBox.autoRange(padding=0.04)
        for name, opts in terminals.items():
            self.addTerminal(name, **opts)

    def setLibrary(self, lib):
        self.library = lib
        self.widget().chartWidget.buildMenu()

    def setInput(self, **args):
        """Set the input values of the flowchart. This will automatically propagate
        the new values throughout the flowchart, (possibly) causing the output to change.
        """
        self.inputWasSet = True
        self.inputNode.setOutput(**args)

    def outputChanged(self):
        vals = self.outputNode.inputValues()
        self.widget().outputChanged(vals)
        self.setOutput(**vals)

    def output(self):
        """Return a dict of the values on the Flowchart's output terminals.
        """
        return self.outputNode.inputValues()

    def nodes(self):
        return self._nodes

    def addTerminal(self, name, **opts):
        term = Node.addTerminal(self, name, **opts)
        name = term.name()
        if opts['io'] == 'in':
            opts['io'] = 'out'
            opts['multi'] = False
            self.inputNode.sigTerminalAdded.disconnect(self.internalTerminalAdded)
            try:
                self.inputNode.addTerminal(name, **opts)
            finally:
                self.inputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        else:
            opts['io'] = 'in'
            self.outputNode.sigTerminalAdded.disconnect(self.internalTerminalAdded)
            try:
                self.outputNode.addTerminal(name, **opts)
            finally:
                self.outputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
        return term

    def removeTerminal(self, name):
        term = self[name]
        inTerm = self.internalTerminal(term)
        Node.removeTerminal(self, name)
        inTerm.node().removeTerminal(inTerm.name())

    def internalTerminalRenamed(self, term, oldName):
        self[oldName].rename(term.name())

    def internalTerminalAdded(self, node, term):
        if term._io == 'in':
            io = 'out'
        else:
            io = 'in'
        Node.addTerminal(self, term.name(), io=io, renamable=term.isRenamable(), removable=term.isRemovable(), multiable=term.isMultiable())

    def internalTerminalRemoved(self, node, term):
        try:
            Node.removeTerminal(self, term.name())
        except KeyError:
            pass

    def terminalRenamed(self, term, oldName):
        newName = term.name()
        Node.terminalRenamed(self, self[oldName], oldName)
        for n in [self.inputNode, self.outputNode]:
            if oldName in n.terminals:
                n[oldName].rename(newName)

    def createNode(self, nodeType, name=None, pos=None):
        """Create a new Node and add it to this flowchart.
        """
        if name is None:
            n = 0
            while True:
                name = '%s.%d' % (nodeType, n)
                if name not in self._nodes:
                    break
                n += 1
        node = self.library.getNodeType(nodeType)(name)
        self.addNode(node, name, pos)
        return node

    def addNode(self, node, name, pos=None):
        """Add an existing Node to this flowchart.
        
        See also: createNode()
        """
        if pos is None:
            pos = [0, 0]
        if type(pos) in [QtCore.QPoint, QtCore.QPointF]:
            pos = [pos.x(), pos.y()]
        item = node.graphicsItem()
        item.setZValue(self.nextZVal * 2)
        self.nextZVal += 1
        self.viewBox.addItem(item)
        item.moveBy(*pos)
        self._nodes[name] = node
        if node is not self.inputNode and node is not self.outputNode:
            self.widget().addNode(node)
        node.sigClosed.connect(self.nodeClosed)
        node.sigRenamed.connect(self.nodeRenamed)
        node.sigOutputChanged.connect(self.nodeOutputChanged)
        self.sigChartChanged.emit(self, 'add', node)

    def removeNode(self, node):
        """Remove a Node from this flowchart.
        """
        node.close()

    def nodeClosed(self, node):
        del self._nodes[node.name()]
        self.widget().removeNode(node)
        for signal, slot in [('sigClosed', self.nodeClosed), ('sigRenamed', self.nodeRenamed), ('sigOutputChanged', self.nodeOutputChanged)]:
            try:
                getattr(node, signal).disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self.sigChartChanged.emit(self, 'remove', node)

    def nodeRenamed(self, node, oldName):
        del self._nodes[oldName]
        self._nodes[node.name()] = node
        if node is not self.inputNode and node is not self.outputNode:
            self.widget().nodeRenamed(node, oldName)
        self.sigChartChanged.emit(self, 'rename', node)

    def arrangeNodes(self):
        pass

    def internalTerminal(self, term):
        """If the terminal belongs to the external Node, return the corresponding internal terminal"""
        if term.node() is self:
            if term.isInput():
                return self.inputNode[term.name()]
            else:
                return self.outputNode[term.name()]
        else:
            return term

    def connectTerminals(self, term1, term2):
        """Connect two terminals together within this flowchart."""
        term1 = self.internalTerminal(term1)
        term2 = self.internalTerminal(term2)
        term1.connectTo(term2)

    def process(self, **args):
        """
        Process data through the flowchart, returning the output.
        
        Keyword arguments must be the names of input terminals. 
        The return value is a dict with one key per output terminal.
        
        """
        data = {}
        order = self.processOrder()
        for n, t in self.inputNode.outputs().items():
            if n in args:
                data[t] = args[n]
        ret = {}
        for c, arg in order:
            if c == 'p':
                node = arg
                if node is self.inputNode:
                    continue
                outs = list(node.outputs().values())
                ins = list(node.inputs().values())
                args = {}
                for inp in ins:
                    inputs = inp.inputTerminals()
                    if len(inputs) == 0:
                        continue
                    if inp.isMultiValue():
                        args[inp.name()] = dict([(i, data[i]) for i in inputs if i in data])
                    else:
                        args[inp.name()] = data[inputs[0]]
                if node is self.outputNode:
                    ret = args
                else:
                    try:
                        if node.isBypassed():
                            result = node.processBypassed(args)
                        else:
                            result = node.process(display=False, **args)
                    except:
                        print('Error processing node %s. Args are: %s' % (str(node), str(args)))
                        raise
                    for out in outs:
                        try:
                            data[out] = result[out.name()]
                        except KeyError:
                            pass
            elif c == 'd':
                if arg in data:
                    del data[arg]
        return ret

    def processOrder(self):
        """Return the order of operations required to process this chart.
        The order returned should look like [('p', node1), ('p', node2), ('d', terminal1), ...] 
        where each tuple specifies either (p)rocess this node or (d)elete the result from this terminal
        """
        deps = {}
        tdeps = {}
        for name, node in self._nodes.items():
            deps[node] = node.dependentNodes()
            for t in node.outputs().values():
                tdeps[t] = t.dependentNodes()
        order = fn.toposort(deps)
        ops = [('p', n) for n in order]
        dels = []
        for t, nodes in tdeps.items():
            lastInd = 0
            lastNode = None
            for n in nodes:
                if n is self:
                    lastInd = None
                    break
                else:
                    try:
                        ind = order.index(n)
                    except ValueError:
                        continue
                if lastNode is None or ind > lastInd:
                    lastNode = n
                    lastInd = ind
            if lastInd is not None:
                dels.append((lastInd + 1, t))
        dels.sort(key=lambda a: a[0], reverse=True)
        for i, t in dels:
            ops.insert(i, ('d', t))
        return ops

    def nodeOutputChanged(self, startNode):
        """Triggered when a node's output values have changed. (NOT called during process())
        Propagates new data forward through network."""
        if self.processing:
            return
        self.processing = True
        try:
            deps = {}
            for name, node in self._nodes.items():
                deps[node] = []
                for t in node.outputs().values():
                    deps[node].extend(t.dependentNodes())
            order = fn.toposort(deps, nodes=[startNode])
            order.reverse()
            terms = set(startNode.outputs().values())
            for node in order[1:]:
                update = False
                for term in list(node.inputs().values()):
                    deps = list(term.connections().keys())
                    for d in deps:
                        if d in terms:
                            update |= True
                            term.inputChanged(d, process=False)
                if update:
                    node.update()
                    terms |= set(node.outputs().values())
        finally:
            self.processing = False
            if self.inputWasSet:
                self.inputWasSet = False
            else:
                self.sigStateChanged.emit()

    def chartGraphicsItem(self):
        """Return the graphicsItem that displays the internal nodes and
        connections of this flowchart.
        
        Note that the similar method `graphicsItem()` is inherited from Node
        and returns the *external* graphical representation of this flowchart."""
        return self.viewBox

    def widget(self):
        """Return the control widget for this flowchart.
        
        This widget provides GUI access to the parameters for each node and a
        graphical representation of the flowchart.
        """
        if self._widget is None:
            self._widget = FlowchartCtrlWidget(self)
            self.scene = self._widget.scene()
            self.viewBox = self._widget.viewBox()
        return self._widget

    def listConnections(self):
        conn = set()
        for n in self._nodes.values():
            terms = n.outputs()
            for t in terms.values():
                for c in t.connections():
                    conn.add((t, c))
        return conn

    def saveState(self):
        """Return a serializable data structure representing the current state of this flowchart. 
        """
        state = Node.saveState(self)
        state['nodes'] = []
        state['connects'] = []
        for name, node in self._nodes.items():
            cls = type(node)
            if hasattr(cls, 'nodeName'):
                clsName = cls.nodeName
                pos = node.graphicsItem().pos()
                ns = {'class': clsName, 'name': name, 'pos': (pos.x(), pos.y()), 'state': node.saveState()}
                state['nodes'].append(ns)
        conn = self.listConnections()
        for a, b in conn:
            state['connects'].append((a.node().name(), a.name(), b.node().name(), b.name()))
        state['inputNode'] = self.inputNode.saveState()
        state['outputNode'] = self.outputNode.saveState()
        return state

    def restoreState(self, state, clear=False):
        """Restore the state of this flowchart from a previous call to `saveState()`.
        """
        self.blockSignals(True)
        try:
            if clear:
                self.clear()
            Node.restoreState(self, state)
            nodes = state['nodes']
            nodes.sort(key=lambda a: a['pos'][0])
            for n in nodes:
                if n['name'] in self._nodes:
                    self._nodes[n['name']].restoreState(n['state'])
                    continue
                try:
                    node = self.createNode(n['class'], name=n['name'])
                    node.restoreState(n['state'])
                except:
                    printExc('Error creating node %s: (continuing anyway)' % n['name'])
            self.inputNode.restoreState(state.get('inputNode', {}))
            self.outputNode.restoreState(state.get('outputNode', {}))
            for n1, t1, n2, t2 in state['connects']:
                try:
                    self.connectTerminals(self._nodes[n1][t1], self._nodes[n2][t2])
                except:
                    print(self._nodes[n1].terminals)
                    print(self._nodes[n2].terminals)
                    printExc('Error connecting terminals %s.%s - %s.%s:' % (n1, t1, n2, t2))
        finally:
            self.blockSignals(False)
        self.outputChanged()
        self.sigChartLoaded.emit()
        self.sigStateChanged.emit()

    def loadFile(self, fileName=None, startDir=None):
        """Load a flowchart (``*.fc``) file.
        """
        if fileName is None:
            if startDir is None:
                startDir = self.filePath
            if startDir is None:
                startDir = '.'
            self.fileDialog = FileDialog(None, 'Load Flowchart..', startDir, 'Flowchart (*.fc)')
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.loadFile)
            return
        state = configfile.readConfigFile(fileName)
        self.restoreState(state, clear=True)
        self.viewBox.autoRange()
        self.sigFileLoaded.emit(fileName)

    def saveFile(self, fileName=None, startDir=None, suggestedFileName='flowchart.fc'):
        """Save this flowchart to a .fc file
        """
        if fileName is None:
            if startDir is None:
                startDir = self.filePath
            if startDir is None:
                startDir = '.'
            self.fileDialog = FileDialog(None, 'Save Flowchart..', startDir, 'Flowchart (*.fc)')
            self.fileDialog.setDefaultSuffix('fc')
            self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.saveFile)
            return
        configfile.writeConfigFile(self.saveState(), fileName)
        self.sigFileSaved.emit(fileName)

    def clear(self):
        """Remove all nodes from this flowchart except the original input/output nodes.
        """
        for n in list(self._nodes.values()):
            if n is self.inputNode or n is self.outputNode:
                continue
            n.close()
        self.widget().clear()

    def clearTerminals(self):
        Node.clearTerminals(self)
        self.inputNode.clearTerminals()
        self.outputNode.clearTerminals()