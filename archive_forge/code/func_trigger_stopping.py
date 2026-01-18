import numpy as np
def trigger_stopping(self, msg, verbose):
    if self.lr_schedule != 'adaptive':
        if verbose:
            print(msg + ' Stopping.')
        return True
    if self.learning_rate <= 1e-06:
        if verbose:
            print(msg + ' Learning rate too small. Stopping.')
        return True
    self.learning_rate /= 5.0
    if verbose:
        print(msg + ' Setting learning rate to %f' % self.learning_rate)
    return False