import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def _try_match(content, prefix, entrypoint):
    code_blocks = [m[1] for m in re.findall('(\\`{3}.*?\\n+)([\\s\\S]*?)(\\n+\\`{3})', content)] + [content]
    for block in code_blocks:
        try:
            code_completion = prefix + block
            if _function_exists(code_completion, entrypoint):
                return code_completion
        except SyntaxError:
            pass