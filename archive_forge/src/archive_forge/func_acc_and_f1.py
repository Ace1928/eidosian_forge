import warnings
from ...utils import is_sklearn_available, requires_backends
def acc_and_f1(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_backends(acc_and_f1, 'sklearn')
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {'acc': acc, 'f1': f1, 'acc_and_f1': (acc + f1) / 2}