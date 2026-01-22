from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
class MaxEntropy:
    """Hold information for a Maximum Entropy classifier.

    Members:
    classes      List of the possible classes of data.
    alphas       List of the weights for each feature.
    feature_fns  List of the feature functions.

    Car data from example Naive Bayes Classifier example by Eric Meisner November 22, 2003
    http://www.inf.u-szeged.hu/~ormandi/teaching

    >>> from Bio.MaxEntropy import train, classify
    >>> xcar = [
    ...     ['Red', 'Sports', 'Domestic'],
    ...     ['Red', 'Sports', 'Domestic'],
    ...     ['Red', 'Sports', 'Domestic'],
    ...     ['Yellow', 'Sports', 'Domestic'],
    ...     ['Yellow', 'Sports', 'Imported'],
    ...     ['Yellow', 'SUV', 'Imported'],
    ...     ['Yellow', 'SUV', 'Imported'],
    ...     ['Yellow', 'SUV', 'Domestic'],
    ...     ['Red', 'SUV', 'Imported'],
    ...     ['Red', 'Sports', 'Imported']]
    >>> ycar = ['Yes','No','Yes','No','Yes','No','Yes','No','No','Yes']

    Requires some rules or features

    >>> def udf1(ts, cl):
    ...     return ts[0] != 'Red'
    ...
    >>> def udf2(ts, cl):
    ...     return ts[1] != 'Sports'
    ...
    >>> def udf3(ts, cl):
    ...     return ts[2] != 'Domestic'
    ...
    >>> user_functions = [udf1, udf2, udf3]  # must be an iterable type
    >>> xe = train(xcar, ycar, user_functions)
    >>> for xv, yv in zip(xcar, ycar):
    ...     xc = classify(xe, xv)
    ...     print('Pred: %s gives %s y is %s' % (xv, xc, yv))
    ...
    Pred: ['Red', 'Sports', 'Domestic'] gives No y is Yes
    Pred: ['Red', 'Sports', 'Domestic'] gives No y is No
    Pred: ['Red', 'Sports', 'Domestic'] gives No y is Yes
    Pred: ['Yellow', 'Sports', 'Domestic'] gives No y is No
    Pred: ['Yellow', 'Sports', 'Imported'] gives No y is Yes
    Pred: ['Yellow', 'SUV', 'Imported'] gives No y is No
    Pred: ['Yellow', 'SUV', 'Imported'] gives No y is Yes
    Pred: ['Yellow', 'SUV', 'Domestic'] gives No y is No
    Pred: ['Red', 'SUV', 'Imported'] gives No y is No
    Pred: ['Red', 'Sports', 'Imported'] gives No y is Yes
    """

    def __init__(self):
        """Initialize the class."""
        self.classes = []
        self.alphas = []
        self.feature_fns = []