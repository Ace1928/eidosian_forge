import pytest
import rpy2.rlike.functional as rlf
@pytest.mark.parametrize('subject_fun', [rlf.iterify, rlf.listify])
def test_simplefunction(subject_fun):

    def f(x):
        return x ** 2
    f_iter = subject_fun(f)
    seq = (1, 2, 3)
    res = f_iter(seq)
    for va, vb in zip(seq, res):
        assert va ** 2 == vb