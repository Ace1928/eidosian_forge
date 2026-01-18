import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def zs_bbh_mc_orca_truthfulqa_orca_match_answer(task_data, response):
    for c in response:
        if c in task_data['options']:
            return (True, c)
    return (False, '')