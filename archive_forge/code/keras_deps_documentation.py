from tensorflow.python.util.tf_export import tf_export
Interface that provides access to Keras dependencies.

This library is a common interface that contains Keras functions needed by
TensorFlow and TensorFlow Lite and is required as per the dependency inversion
principle (https://en.wikipedia.org/wiki/Dependency_inversion_principle). As per
this principle, high-level modules (eg: TensorFlow and TensorFlow Lite) should
not depend on low-level modules (eg: Keras) and instead both should depend on a
common interface such as this file.
