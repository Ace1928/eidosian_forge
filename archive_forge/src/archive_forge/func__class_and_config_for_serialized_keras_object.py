import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _class_and_config_for_serialized_keras_object(config, module_objects=None, custom_objects=None, printable_module_name='object'):
    """Returns the class name and config for a serialized keras object."""
    if not isinstance(config, dict) or 'class_name' not in config or 'config' not in config:
        raise ValueError('Improper config format: ' + str(config))
    class_name = config['class_name']
    cls = _get_registered_object(class_name, custom_objects=custom_objects, module_objects=module_objects)
    if cls is None:
        raise ValueError('Unknown ' + printable_module_name + ': ' + class_name)
    cls_config = config['config']
    deserialized_objects = {}
    for key, item in cls_config.items():
        if isinstance(item, dict) and '__passive_serialization__' in item:
            deserialized_objects[key] = _deserialize_keras_object(item, module_objects=module_objects, custom_objects=custom_objects, printable_module_name='config_item')
        elif isinstance(item, six.string_types) and tf_inspect.isfunction(_get_registered_object(item, custom_objects)):
            deserialized_objects[key] = _get_registered_object(item, custom_objects)
    for key, item in deserialized_objects.items():
        cls_config[key] = deserialized_objects[key]
    return (cls, cls_config)