import deap
from copy import copy
def _combine_dfs(left, right, operators):

    def _make_branch(branch):
        if branch == 'input_matrix':
            return 'FunctionTransformer(copy)'
        elif branch[0] == 'CombineDFs':
            return _combine_dfs(branch[1], branch[2], operators)
        elif branch[1] == 'input_matrix':
            tpot_op = get_by_name(branch[0], operators)
            if tpot_op.root:
                return 'StackingEstimator(estimator={})'.format(_process_operator(branch, operators)[0])
            else:
                return _process_operator(branch, operators)[0]
        else:
            tpot_op = get_by_name(branch[0], operators)
            if tpot_op.root:
                return 'StackingEstimator(estimator={})'.format(generate_pipeline_code(branch, operators))
            else:
                return generate_pipeline_code(branch, operators)
    return 'make_union(\n{},\n{}\n)'.format(_indent(_make_branch(left), 4), _indent(_make_branch(right), 4))