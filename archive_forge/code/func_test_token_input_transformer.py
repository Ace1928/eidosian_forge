import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_token_input_transformer():
    tests = [('1.2', "Decimal ('1.2')"), ('"1.2"', '"1.2"')]
    tt.check_pairs(transform_and_reset(decistmt), tests)
    ml_tests = [[("a = 1.2; b = '''x", None), ("y'''", "a =Decimal ('1.2');b ='''x\ny'''")], [('a = [1.2,', None), ('3]', "a =[Decimal ('1.2'),\n3 ]")], [("a = '''foo", None), ('bar', None), (None, "a = '''foo\nbar")]]
    for example in ml_tests:
        transform_checker(example, decistmt)