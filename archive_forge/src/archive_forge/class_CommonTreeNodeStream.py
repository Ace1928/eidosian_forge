from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import UP, DOWN, EOF, INVALID_TOKEN_TYPE
from antlr3.exceptions import MismatchedTreeNodeException, \
from antlr3.recognizers import BaseRecognizer, RuleReturnScope
from antlr3.streams import IntStream
from antlr3.tokens import CommonToken, Token, INVALID_TOKEN
import six
from six.moves import range
class CommonTreeNodeStream(TreeNodeStream):
    """@brief A buffered stream of tree nodes.

    Nodes can be from a tree of ANY kind.

    This node stream sucks all nodes out of the tree specified in
    the constructor during construction and makes pointers into
    the tree using an array of Object pointers. The stream necessarily
    includes pointers to DOWN and UP and EOF nodes.

    This stream knows how to mark/release for backtracking.

    This stream is most suitable for tree interpreters that need to
    jump around a lot or for tree parsers requiring speed (at cost of memory).
    There is some duplicated functionality here with UnBufferedTreeNodeStream
    but just in bookkeeping, not tree walking etc...

    @see UnBufferedTreeNodeStream
    """

    def __init__(self, *args):
        TreeNodeStream.__init__(self)
        if len(args) == 1:
            adaptor = CommonTreeAdaptor()
            tree = args[0]
        elif len(args) == 2:
            adaptor = args[0]
            tree = args[1]
        else:
            raise TypeError('Invalid arguments')
        self.down = adaptor.createFromType(DOWN, 'DOWN')
        self.up = adaptor.createFromType(UP, 'UP')
        self.eof = adaptor.createFromType(EOF, 'EOF')
        self.nodes = []
        self.root = tree
        self.tokens = None
        self.adaptor = adaptor
        self.uniqueNavigationNodes = False
        self.p = -1
        self.lastMarker = None
        self.calls = []

    def fillBuffer(self):
        """Walk tree with depth-first-search and fill nodes buffer.

        Don't do DOWN, UP nodes if its a list (t is isNil).
        """
        self._fillBuffer(self.root)
        self.p = 0

    def _fillBuffer(self, t):
        nil = self.adaptor.isNil(t)
        if not nil:
            self.nodes.append(t)
        n = self.adaptor.getChildCount(t)
        if not nil and n > 0:
            self.addNavigationNode(DOWN)
        for c in range(n):
            self._fillBuffer(self.adaptor.getChild(t, c))
        if not nil and n > 0:
            self.addNavigationNode(UP)

    def getNodeIndex(self, node):
        """What is the stream index for node?

    0..n-1
        Return -1 if node not found.
        """
        if self.p == -1:
            self.fillBuffer()
        for i, t in enumerate(self.nodes):
            if t == node:
                return i
        return -1

    def addNavigationNode(self, ttype):
        """
        As we flatten the tree, we use UP, DOWN nodes to represent
        the tree structure.  When debugging we need unique nodes
        so instantiate new ones when uniqueNavigationNodes is true.
        """
        navNode = None
        if ttype == DOWN:
            if self.hasUniqueNavigationNodes():
                navNode = self.adaptor.createFromType(DOWN, 'DOWN')
            else:
                navNode = self.down
        elif self.hasUniqueNavigationNodes():
            navNode = self.adaptor.createFromType(UP, 'UP')
        else:
            navNode = self.up
        self.nodes.append(navNode)

    def get(self, i):
        if self.p == -1:
            self.fillBuffer()
        return self.nodes[i]

    def LT(self, k):
        if self.p == -1:
            self.fillBuffer()
        if k == 0:
            return None
        if k < 0:
            return self.LB(-k)
        if self.p + k - 1 >= len(self.nodes):
            return self.eof
        return self.nodes[self.p + k - 1]

    def getCurrentSymbol(self):
        return self.LT(1)

    def LB(self, k):
        """Look backwards k nodes"""
        if k == 0:
            return None
        if self.p - k < 0:
            return None
        return self.nodes[self.p - k]

    def getTreeSource(self):
        return self.root

    def getSourceName(self):
        return self.getTokenStream().getSourceName()

    def getTokenStream(self):
        return self.tokens

    def setTokenStream(self, tokens):
        self.tokens = tokens

    def getTreeAdaptor(self):
        return self.adaptor

    def hasUniqueNavigationNodes(self):
        return self.uniqueNavigationNodes

    def setUniqueNavigationNodes(self, uniqueNavigationNodes):
        self.uniqueNavigationNodes = uniqueNavigationNodes

    def consume(self):
        if self.p == -1:
            self.fillBuffer()
        self.p += 1

    def LA(self, i):
        return self.adaptor.getType(self.LT(i))

    def mark(self):
        if self.p == -1:
            self.fillBuffer()
        self.lastMarker = self.index()
        return self.lastMarker

    def release(self, marker=None):
        pass

    def index(self):
        return self.p

    def rewind(self, marker=None):
        if marker is None:
            marker = self.lastMarker
        self.seek(marker)

    def seek(self, index):
        if self.p == -1:
            self.fillBuffer()
        self.p = index

    def push(self, index):
        """
        Make stream jump to a new location, saving old location.
        Switch back with pop().
        """
        self.calls.append(self.p)
        self.seek(index)

    def pop(self):
        """
        Seek back to previous index saved during last push() call.
        Return top of stack (return index).
        """
        ret = self.calls.pop(-1)
        self.seek(ret)
        return ret

    def reset(self):
        self.p = 0
        self.lastMarker = 0
        self.calls = []

    def size(self):
        if self.p == -1:
            self.fillBuffer()
        return len(self.nodes)

    def replaceChildren(self, parent, startChildIndex, stopChildIndex, t):
        if parent is not None:
            self.adaptor.replaceChildren(parent, startChildIndex, stopChildIndex, t)

    def __str__(self):
        """Used for testing, just return the token type stream"""
        if self.p == -1:
            self.fillBuffer()
        return ' '.join([str(self.adaptor.getType(node)) for node in self.nodes])

    def toString(self, start, stop):
        if start is None or stop is None:
            return None
        if self.p == -1:
            self.fillBuffer()
        if self.tokens is not None:
            beginTokenIndex = self.adaptor.getTokenStartIndex(start)
            endTokenIndex = self.adaptor.getTokenStopIndex(stop)
            if self.adaptor.getType(stop) == UP:
                endTokenIndex = self.adaptor.getTokenStopIndex(start)
            elif self.adaptor.getType(stop) == EOF:
                endTokenIndex = self.size() - 2
            return self.tokens.toString(beginTokenIndex, endTokenIndex)
        i, t = (0, None)
        for i, t in enumerate(self.nodes):
            if t == start:
                break
        buf = []
        t = self.nodes[i]
        while t != stop:
            text = self.adaptor.getText(t)
            if text is None:
                text = ' ' + self.adaptor.getType(t)
            buf.append(text)
            i += 1
            t = self.nodes[i]
        text = self.adaptor.getText(stop)
        if text is None:
            text = ' ' + self.adaptor.getType(stop)
        buf.append(text)
        return ''.join(buf)

    def __iter__(self):
        if self.p == -1:
            self.fillBuffer()
        for node in self.nodes:
            yield node