import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
class PythonMongoishQueryGenerator:
    SPACER = '----------'
    DECIMAL_SPACER = ';;;'
    FRONTEND_NAME_MAPPING = {'ID': 'name', 'Name': 'displayName', 'Tags': 'tags', 'State': 'state', 'CreatedTimestamp': 'createdAt', 'Runtime': 'duration', 'User': 'username', 'Sweep': 'sweep', 'Group': 'group', 'JobType': 'jobType', 'Hostname': 'host', 'UsingArtifact': 'inputArtifacts', 'OutputtingArtifact': 'outputArtifacts', 'Step': '_step', 'Relative Time (Wall)': '_absolute_runtime', 'Relative Time (Process)': '_runtime', 'Wall Time': '_timestamp'}
    FRONTEND_NAME_MAPPING_REVERSED = {v: k for k, v in FRONTEND_NAME_MAPPING.items()}
    AST_OPERATORS = {ast.Lt: '$lt', ast.LtE: '$lte', ast.Gt: '$gt', ast.GtE: '$gte', ast.Eq: '=', ast.Is: '=', ast.NotEq: '$ne', ast.IsNot: '$ne', ast.In: '$in', ast.NotIn: '$nin', ast.And: '$and', ast.Or: '$or', ast.Not: '$not'}
    if sys.version_info >= (3, 8):
        AST_FIELDS = {ast.Constant: 'value', ast.Name: 'id', ast.List: 'elts', ast.Tuple: 'elts'}
    else:
        AST_FIELDS = {ast.Str: 's', ast.Num: 'n', ast.Name: 'id', ast.List: 'elts', ast.Tuple: 'elts', ast.NameConstant: 'value'}

    def __init__(self, run_set):
        self.run_set = run_set
        self.panel_metrics_helper = PanelMetricsHelper()

    def _handle_compare(self, node):
        left = self.front_to_back(self._handle_fields(node.left))
        op = self._handle_ops(node.ops[0])
        right = self._handle_fields(node.comparators[0])
        if op == '=':
            return {left: right}
        else:
            return {left: {op: right}}

    def _handle_fields(self, node):
        result = getattr(node, self.AST_FIELDS.get(type(node)))
        if isinstance(result, list):
            return [self._handle_fields(node) for node in result]
        elif isinstance(result, str):
            return self._unconvert(result)
        return result

    def _handle_ops(self, node):
        return self.AST_OPERATORS.get(type(node))

    def _replace_numeric_dots(self, s):
        numeric_dots = []
        for i, (left, mid, right) in enumerate(zip(s, s[1:], s[2:]), 1):
            if mid == '.':
                if left.isdigit() and right.isdigit() or (left.isdigit() and right == ' ') or (left == ' ' and right.isdigit()):
                    numeric_dots.append(i)
        if s[-2].isdigit() and s[-1] == '.':
            numeric_dots.append(len(s) - 1)
        numeric_dots = [-1] + numeric_dots + [len(s)]
        substrs = []
        for start, stop in zip(numeric_dots, numeric_dots[1:]):
            substrs.append(s[start + 1:stop])
            substrs.append(self.DECIMAL_SPACER)
        substrs = substrs[:-1]
        return ''.join(substrs)

    def _convert(self, filterstr):
        _conversion = self._replace_numeric_dots(filterstr).replace('.', self.SPACER).replace(self.DECIMAL_SPACER, '.')
        return '(' + _conversion + ')'

    def _unconvert(self, field_name):
        return field_name.replace(self.SPACER, '.')

    def python_to_mongo(self, filterstr):
        try:
            tree = ast.parse(self._convert(filterstr), mode='eval')
        except SyntaxError as e:
            raise ValueError('Invalid python comparison expression; form something like `my_col == 123`') from e
        multiple_filters = hasattr(tree.body, 'op')
        if multiple_filters:
            op = self.AST_OPERATORS.get(type(tree.body.op))
            values = [self._handle_compare(v) for v in tree.body.values]
        else:
            op = '$and'
            values = [self._handle_compare(tree.body)]
        return {'$or': [{op: values}]}

    def front_to_back(self, name):
        name, *rest = name.split('.')
        rest = '.' + '.'.join(rest) if rest else ''
        if name in self.FRONTEND_NAME_MAPPING:
            return self.FRONTEND_NAME_MAPPING[name]
        elif name in self.FRONTEND_NAME_MAPPING_REVERSED:
            return name
        elif name in self.run_set._runs_config:
            return f'config.{name}.value{rest}'
        else:
            return f'summary_metrics.{name}{rest}'

    def back_to_front(self, name):
        if name in self.FRONTEND_NAME_MAPPING_REVERSED:
            return self.FRONTEND_NAME_MAPPING_REVERSED[name]
        elif name in self.FRONTEND_NAME_MAPPING:
            return name
        elif name.startswith('config.') and '.value' in name:
            return name.replace('config.', '').replace('.value', '')
        elif name.startswith('summary_metrics.'):
            return name.replace('summary_metrics.', '')
        wandb.termerror(f'Unknown token: {name}')
        return name

    def pc_front_to_back(self, name):
        name, *rest = name.split('.')
        rest = '.' + '.'.join(rest) if rest else ''
        if name is None:
            return None
        elif name in self.panel_metrics_helper.FRONTEND_NAME_MAPPING:
            return 'summary:' + self.panel_metrics_helper.FRONTEND_NAME_MAPPING[name]
        elif name in self.FRONTEND_NAME_MAPPING:
            return self.FRONTEND_NAME_MAPPING[name]
        elif name in self.FRONTEND_NAME_MAPPING_REVERSED:
            return name
        elif name in self.run_set._runs_config:
            return f'config:{name}.value{rest}'
        else:
            return f'summary:{name}{rest}'

    def pc_back_to_front(self, name):
        if name is None:
            return None
        elif 'summary:' in name:
            name = name.replace('summary:', '')
            return self.panel_metrics_helper.FRONTEND_NAME_MAPPING_REVERSED.get(name, name)
        elif name in self.FRONTEND_NAME_MAPPING_REVERSED:
            return self.FRONTEND_NAME_MAPPING_REVERSED[name]
        elif name in self.FRONTEND_NAME_MAPPING:
            return name
        elif name.startswith('config:') and '.value' in name:
            return name.replace('config:', '').replace('.value', '')
        elif name.startswith('summary_metrics.'):
            return name.replace('summary_metrics.', '')
        return name