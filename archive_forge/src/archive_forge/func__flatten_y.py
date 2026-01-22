import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
def _flatten_y(self, y):
    if isinstance(y, dict) and self.output_names:
        result = []
        for name in self.output_names:
            if name in y:
                result.append(y[name])
        return result
    return tree.flatten(y)