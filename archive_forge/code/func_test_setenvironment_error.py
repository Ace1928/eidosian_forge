import pytest
import rpy2.robjects as robjects
def test_setenvironment_error():
    fml = robjects.Formula('y ~ x')
    with pytest.raises(TypeError):
        fml.setenvironment(rinterface.NA_Logical)