from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyCy(CythonCommand):
    """
    Invoke a Cython command. Available commands are:

        cy import
        cy break
        cy step
        cy next
        cy run
        cy cont
        cy finish
        cy up
        cy down
        cy select
        cy bt / cy backtrace
        cy list
        cy print
        cy set
        cy locals
        cy globals
        cy exec
    """
    name = 'cy'
    command_class = gdb.COMMAND_NONE
    completer_class = gdb.COMPLETE_COMMAND

    def __init__(self, name, command_class, completer_class):
        super(CythonCommand, self).__init__(name, command_class, completer_class, prefix=True)
        commands = dict(import_=CyImport.register(), break_=CyBreak.register(), step=CyStep.register(), next=CyNext.register(), run=CyRun.register(), cont=CyCont.register(), finish=CyFinish.register(), up=CyUp.register(), down=CyDown.register(), select=CySelect.register(), bt=CyBacktrace.register(), list=CyList.register(), print_=CyPrint.register(), locals=CyLocals.register(), globals=CyGlobals.register(), exec_=libpython.FixGdbCommand('cy exec', '-cy-exec'), _exec=CyExec.register(), set=CySet.register(), cy_cname=CyCName('cy_cname'), cy_cvalue=CyCValue('cy_cvalue'), cy_lineno=CyLine('cy_lineno'), cy_eval=CyEval('cy_eval'))
        for command_name, command in commands.items():
            command.cy = self
            setattr(self, command_name, command)
        self.cy = self
        self.cython_namespace = {}
        self.functions_by_qualified_name = {}
        self.functions_by_cname = {}
        self.functions_by_name = collections.defaultdict(list)