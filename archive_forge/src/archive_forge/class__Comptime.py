import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
class _Comptime:

    @staticmethod
    def __call__(fn):
        """fn gets called at compile time in TorchDynamo, does nothing otherwise"""
        return

    @staticmethod
    def graph_break():
        comptime(lambda ctx: ctx.graph_break())

    @staticmethod
    def print_graph():
        comptime(lambda ctx: ctx.print_graph())

    @staticmethod
    def print_disas(*, stacklevel=0):
        comptime(lambda ctx: ctx.print_disas(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_value_stack(*, stacklevel=0):
        comptime(lambda ctx: ctx.print_value_stack(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_value_stack_and_return(e, *, stacklevel=0):
        comptime(lambda ctx: ctx.print_value_stack(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))
        return e

    @staticmethod
    def print_locals(*, stacklevel=0):
        comptime(lambda ctx: ctx.print_locals(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_bt(*, stacklevel=0):
        comptime(lambda ctx: ctx.print_bt(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_guards():
        comptime(lambda ctx: ctx.print_guards())

    @staticmethod
    def breakpoint():
        """
        Like pdb breakpoint(), but drop into pdb whenever this line
        of code is compiled by dynamo.  Use it by putting
        this in your model code::

            from torch._dynamo.comptime import comptime
            comptime.breakpoint()

        And then, inside pdb, you can access 'ctx' to query things
        about the compilation context::

            (Pdb) !ctx.print_bt()
            (Pdb) !ctx.print_locals()
            (Pdb) p ctx.get_local("attention").as_fake()
        """

        def inner(inner_ctx):
            ctx = inner_ctx.parent()
            builtins.breakpoint()
        comptime(inner)