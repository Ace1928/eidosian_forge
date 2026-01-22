import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput
class ReplWidget(QtWidgets.QWidget):
    sigCommandEntered = QtCore.Signal(object, object)
    sigCommandRaisedException = QtCore.Signal(object, object)

    def __init__(self, globals, locals, parent=None):
        self.globals = globals
        self.locals = locals
        self._lastCommandRow = None
        self._commandBuffer = []
        self.stdoutInterceptor = StdoutInterceptor(self.write)
        self.ps1 = '>>> '
        self.ps2 = '... '
        QtWidgets.QWidget.__init__(self, parent=parent)
        self._setupUi()
        isDark = self.output.palette().color(QtGui.QPalette.ColorRole.Base).value() < 128
        outputBlockFormat = QtGui.QTextBlockFormat()
        outputFirstLineBlockFormat = QtGui.QTextBlockFormat(outputBlockFormat)
        outputFirstLineBlockFormat.setTopMargin(5)
        outputCharFormat = QtGui.QTextCharFormat()
        outputCharFormat.setFontWeight(QtGui.QFont.Weight.Normal)
        cmdBlockFormat = QtGui.QTextBlockFormat()
        cmdBlockFormat.setBackground(mkBrush('#335' if isDark else '#CCF'))
        cmdCharFormat = QtGui.QTextCharFormat()
        cmdCharFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        self.textStyles = {'command': (cmdCharFormat, cmdBlockFormat), 'output': (outputCharFormat, outputBlockFormat), 'output_first_line': (outputCharFormat, outputFirstLineBlockFormat)}
        self.input.ps1 = self.ps1
        self.input.ps2 = self.ps2

    def _setupUi(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.output = QtWidgets.QTextEdit(self)
        font = QtGui.QFont()
        font.setFamily('Courier New')
        font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
        self.output.setFont(font)
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)
        self.inputWidget = QtWidgets.QWidget(self)
        self.layout.addWidget(self.inputWidget)
        self.inputLayout = QtWidgets.QHBoxLayout()
        self.inputWidget.setLayout(self.inputLayout)
        self.inputLayout.setContentsMargins(0, 0, 0, 0)
        self.input = CmdInput(parent=self)
        self.inputLayout.addWidget(self.input)
        self.input.sigExecuteCmd.connect(self.runCmd)

    def runCmd(self, cmd):
        if '\n' in cmd:
            for line in cmd.split('\n'):
                self.runCmd(line)
            return
        if len(self._commandBuffer) == 0:
            self.write(f'{self.ps1}{cmd}\n', style='command')
        else:
            self.write(f'{self.ps2}{cmd}\n', style='command')
        self.sigCommandEntered.emit(self, cmd)
        self._commandBuffer.append(cmd)
        fullcmd = '\n'.join(self._commandBuffer)
        try:
            cmdCode = code.compile_command(fullcmd)
            self.input.setMultiline(False)
        except Exception:
            self._commandBuffer = []
            self.displayException()
            self.input.setMultiline(False)
        else:
            if cmdCode is None:
                self.input.setMultiline(True)
                return
            self._commandBuffer = []
            try:
                with self.stdoutInterceptor:
                    exec(cmdCode, self.globals(), self.locals())
            except Exception as exc:
                self.displayException()
                self.sigCommandRaisedException.emit(self, exc)
            cursor = self.output.textCursor()
            if cursor.columnNumber() > 0:
                self.write('\n')

    def write(self, strn, style='output', scrollToBottom='auto'):
        """Write a string into the console.

        If scrollToBottom is 'auto', then the console is automatically scrolled
        to fit the new text only if it was already at the bottom.
        """
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            sys.__stdout__.write(strn)
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.output.setTextCursor(cursor)
        sb = self.output.verticalScrollBar()
        scroll = sb.value()
        if scrollToBottom == 'auto':
            atBottom = scroll == sb.maximum()
            scrollToBottom = atBottom
        row = cursor.blockNumber()
        if style == 'command':
            self._lastCommandRow = row
        if style == 'output' and row == self._lastCommandRow + 1:
            firstLine, endl, strn = strn.partition('\n')
            self._setTextStyle('output_first_line')
            self.output.insertPlainText(firstLine + endl)
        if len(strn) > 0:
            self._setTextStyle(style)
            self.output.insertPlainText(strn)
            if style != 'output':
                self._setTextStyle('output')
        if scrollToBottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scroll)

    def displayException(self):
        """
        Display the current exception and stack.
        """
        tb = traceback.format_exc()
        lines = []
        indent = 4
        prefix = ''
        for l in tb.split('\n'):
            lines.append(' ' * indent + prefix + l)
        self.write('\n'.join(lines))

    def _setTextStyle(self, style):
        charFormat, blockFormat = self.textStyles[style]
        cursor = self.output.textCursor()
        cursor.setBlockFormat(blockFormat)
        self.output.setCurrentCharFormat(charFormat)