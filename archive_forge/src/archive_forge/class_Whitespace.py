import os
import sys
import tokenize
class Whitespace:
    S, T = ' \t'

    def __init__(self, ws):
        self.raw = ws
        S, T = (Whitespace.S, Whitespace.T)
        count = []
        b = n = nt = 0
        for ch in self.raw:
            if ch == S:
                n = n + 1
                b = b + 1
            elif ch == T:
                n = n + 1
                nt = nt + 1
                if b >= len(count):
                    count = count + [0] * (b - len(count) + 1)
                count[b] = count[b] + 1
                b = 0
            else:
                break
        self.n = n
        self.nt = nt
        self.norm = (tuple(count), b)
        self.is_simple = len(count) <= 1

    def longest_run_of_spaces(self):
        count, trailing = self.norm
        return max(len(count) - 1, trailing)

    def indent_level(self, tabsize):
        count, trailing = self.norm
        il = 0
        for i in range(tabsize, len(count)):
            il = il + i // tabsize * count[i]
        return trailing + tabsize * (il + self.nt)

    def equal(self, other):
        return self.norm == other.norm

    def not_equal_witness(self, other):
        n = max(self.longest_run_of_spaces(), other.longest_run_of_spaces()) + 1
        a = []
        for ts in range(1, n + 1):
            if self.indent_level(ts) != other.indent_level(ts):
                a.append((ts, self.indent_level(ts), other.indent_level(ts)))
        return a

    def less(self, other):
        if self.n >= other.n:
            return False
        if self.is_simple and other.is_simple:
            return self.nt <= other.nt
        n = max(self.longest_run_of_spaces(), other.longest_run_of_spaces()) + 1
        for ts in range(2, n + 1):
            if self.indent_level(ts) >= other.indent_level(ts):
                return False
        return True

    def not_less_witness(self, other):
        n = max(self.longest_run_of_spaces(), other.longest_run_of_spaces()) + 1
        a = []
        for ts in range(1, n + 1):
            if self.indent_level(ts) >= other.indent_level(ts):
                a.append((ts, self.indent_level(ts), other.indent_level(ts)))
        return a