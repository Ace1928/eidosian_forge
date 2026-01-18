import unittest
import onnx
from onnx import checker, utils
#   1. build a model with graph below. extract models with output combinations
        #   2. validate extracted models' local functions
        #
        # model graph:
        #      i0                    i1                 i2
        #      |   __________________|__________________/_________
        #      |  |                  |             |   /          |
        #      |  |                  |             |  /           |
        #   func_add        func_identity          add         identity
        #    |  ___\___________\____________________|_________    |
        #    | |    \           \                   |  _______|___|
        #    | |     \           \                  | |       |   |
        #    add     function_nested_identity_add   add     function_nested_identity_add
        #     |                 |                    |              |
        #     |                 |                    |              |
        #   o_func_add      o_all_func0           o_no_func     o_all_func1
        #
        # where function_nested_identity_add is a function that is defined with functions:
        #       a               b
        #       |               |
        #   func_identity   func_identity
        #             \       /
        #             func_add
        #                |
        #                c
        #
        