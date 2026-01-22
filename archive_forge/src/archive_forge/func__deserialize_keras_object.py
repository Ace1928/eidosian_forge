import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _deserialize_keras_object(identifier, module_objects=None, custom_objects=None, printable_module_name='object'):
    """Turns the serialized form of a Keras object back into an actual object."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        config = identifier
        cls, cls_config = _class_and_config_for_serialized_keras_object(config, module_objects, custom_objects, printable_module_name)
        if hasattr(cls, 'from_config'):
            arg_spec = tf_inspect.getfullargspec(cls.from_config)
            custom_objects = custom_objects or {}
            if 'custom_objects' in arg_spec.args:
                return cls.from_config(cls_config, custom_objects=dict(list(custom_objects.items())))
            return cls.from_config(cls_config)
        else:
            custom_objects = custom_objects or {}
            return cls(**cls_config)
    elif isinstance(identifier, six.string_types):
        object_name = identifier
        if custom_objects and object_name in custom_objects:
            obj = custom_objects.get(object_name)
        else:
            obj = module_objects.get(object_name)
            if obj is None:
                raise ValueError('Unknown ' + printable_module_name + ': ' + object_name)
        if tf_inspect.isclass(obj):
            return obj()
        return obj
    elif tf_inspect.isfunction(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret serialized %s: %s' % (printable_module_name, identifier))