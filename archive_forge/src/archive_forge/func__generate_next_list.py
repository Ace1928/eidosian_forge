import math
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer
def _generate_next_list(current_list, n):
    new_list = []
    for item in current_list:
        temp_1 = [number * 3 for number in item if number * 3 <= n]
        temp_2 = [number * 3 - 1 for number in item if number * 3 - 1 <= n]
        new_item = temp_1 + temp_2
        new_list.append(new_item)
    last_list = [3 * k + 1 for k in range(len(current_list) + 1) if 3 * k + 1 <= n]
    new_list.append(last_list)
    current_list = new_list
    return current_list