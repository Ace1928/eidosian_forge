from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def CalcSimpleDescriptorsForComposition(self, compos='', composList=None):
    """ calculates all simple descriptors for a given composition

      **Arguments**

        - compos: a string representation of the composition

        - composList: a *composVect*

        The client must provide either _compos_ or _composList_.  If both are
        provided, _composList_ takes priority.

      **Returns**
        the list of descriptor values

      **Notes**

        - when _compos_ is provided, this uses _chemutils.SplitComposition_
          to split the composition into its individual pieces

        - if problems are encountered because of either an unknown descriptor or
          atom type, a _KeyError_ will be raised.

    """
    if composList is None:
        composList = chemutils.SplitComposition(compos)
    try:
        res = []
        for descName, targets in self.simpleList:
            for target in targets:
                try:
                    method = getattr(self, target)
                except AttributeError:
                    print('Method %s does not exist' % target)
                else:
                    res.append(method(descName, composList))
    except KeyError as msg:
        print('composition %s caused problems' % composList)
        raise KeyError(msg)
    return res