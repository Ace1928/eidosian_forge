from taskflow import engines
from taskflow.engines.action_engine import compiler
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils as test_utils
def test_wrong_object(self):
    msg_regex = '^Unknown object .* requested to compile'
    self.assertRaisesRegex(TypeError, msg_regex, compiler.PatternCompiler(42).compile)