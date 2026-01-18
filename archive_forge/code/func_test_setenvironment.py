import pytest
import rpy2.robjects as robjects
def test_setenvironment():
    fml = robjects.Formula('y ~ x')
    newenv = robjects.baseenv['new.env']()
    env = fml.getenvironment()
    assert not newenv.rsame(env)
    fml.setenvironment(newenv)
    env = fml.getenvironment()
    assert newenv.rsame(env)