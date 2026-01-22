import tree
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
def __jax_array__(self):
    raise ValueError('A KerasTensor cannot be used as input to a JAX function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:\n\n```\nx = Input(...)\n...\njax_fn(x)  # Invalid.\n```\n\nWhat you should do instead is wrap `jax_fn` in a layer:\n\n```\nclass MyLayer(Layer):\n    def call(self, x):\n        return jax_fn(x)\n\nx = MyLayer()(x)\n```\n')