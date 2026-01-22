import sys
from pyasn1.type import error
class InnerTypeConstraint(AbstractConstraint):
    """Value must satisfy the type and presence constraints"""

    def _testValue(self, value, idx):
        if self.__singleTypeConstraint:
            self.__singleTypeConstraint(value)
        elif self.__multipleTypeConstraint:
            if idx not in self.__multipleTypeConstraint:
                raise error.ValueConstraintError(value)
            constraint, status = self.__multipleTypeConstraint[idx]
            if status == 'ABSENT':
                raise error.ValueConstraintError(value)
            constraint(value)

    def _setValues(self, values):
        self.__multipleTypeConstraint = {}
        self.__singleTypeConstraint = None
        for v in values:
            if isinstance(v, tuple):
                self.__multipleTypeConstraint[v[0]] = (v[1], v[2])
            else:
                self.__singleTypeConstraint = v
        AbstractConstraint._setValues(self, values)