import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_augAssign__(self, op, v, i):
    if op == '+=':
        return self.__rl_add__(v, i)
    if op == '-=':
        return v - i
    if op == '*=':
        return self.__rl_mult__(v, i)
    if op == '/=':
        return v / i
    if op == '%=':
        return v % i
    if op == '**=':
        return self.__rl_pow__(v, i)
    if op == '<<=':
        return v << i
    if op == '>>=':
        return v >> i
    if op == '|=':
        return v | i
    if op == '^=':
        return v ^ i
    if op == '&=':
        return v & i
    if op == '//=':
        return v // i