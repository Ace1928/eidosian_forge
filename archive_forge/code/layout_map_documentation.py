import collections
import contextlib
import re
import threading
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import lazy_variable
from keras.src.dtensor import utils
from keras.src.engine import base_layer
from tensorflow.python.util.tf_export import keras_export
Apply layout to all `tf.Variable` instances created under the scope.

        All `tf.Variable` instances created under this scope
        will be lazily initialized first. Once they are attached as the model
        or layer attributes, and there is a stable layout mapping for it, the
        variables will be reinitialized into a
        `tf.experimental.dtensor.DVariable` with corresponding layout.

        Note that the layout mapping will use object/attribute names as the
        keys to map the variable to the layout.

        For subclassed models, the full object/attribute name is used as the
        key. For Functional/Sequential models, we use `layer.name` as
        the key for the layer, followed by the attribute name. Keras ensures
        name uniqueness among the layers within a Functional/Sequential model.

        See the following examples that show variable object names
        for different Keras model types:

        ```python
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        layout_map['d1.kernel'] = layout_1
        layout_map['d1.bias'] = layout_2
        layout_map['d2.kernel'] = layout_3
        layout_map['d2.bias'] = layout_4

        ## Subclassed model
        class SubclassModel(tf.keras.Model):

          def __init__(self, name=None):
            super().__init__(name=name)
            self.d1 = tf.keras.layers.Dense(1000)
            self.d2 = tf.keras.layers.Dense(1000)

          def call(self, inputs):
            x = self.d1(inputs)
            return self.d2(x)

        with layout_map.scope():
          model = SubclassModel()
        inputs = tf.zeros((10, 10))
        results = model(inputs)

        model.d1.kernel.layout == layout_1
        model.d1.bias.layout == layout_2
        model.d2.kernel.layout == layout_3
        model.d2.bias.layout == layout_4

        ## Functional model
        with layout_map.scope():
          inputs = tf.keras.Input((10,), batch_size=10)
          x = tf.keras.layers.Dense(20, name='d1')(inputs)
          output = tf.keras.layers.Dense(30, name='d2')(x)

          model = tf.keras.Model(inputs, output)

        d1 = model.layers[1]
        d2 = model.layers[2]

        d1.kernel.layout == layout_1
        d1.bias.layout == layout_2
        d1.kernel.layout == layout_3
        d1.bias.layout == layout_4

        ## Sequential model
        with layout_map.scope():
          model = tf.keras.Sequential([
              tf.keras.layers.Dense(20, name='d1', input_shape=(10,)),
              tf.keras.layers.Dense(30, name='d2')
          ])

        d1 = model.layers[0]
        d2 = model.layers[1]

        d1.kernel.layout == layout_1
        d1.bias.layout == layout_2
        d1.kernel.layout == layout_3
        d1.bias.layout == layout_4
        ```

        Returns:
          A context that will lazily initialize all `tf.Variable` objects
          within the model, with their attributed layouts.
        