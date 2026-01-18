import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def zs_agieval_match_answer(task_data, response):
    letter_set = {'A', 'B', 'C', 'D', 'E', 'F'}
    for c in response:
        if c in letter_set:
            return (True, c)
    return (False, '')