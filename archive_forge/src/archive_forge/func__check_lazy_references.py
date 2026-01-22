import inspect
import types
from collections import defaultdict
from itertools import chain
from django.apps import apps
from django.conf import settings
from django.core.checks import Error, Tags, Warning, register
def _check_lazy_references(apps, ignore=None):
    """
    Ensure all lazy (i.e. string) model references have been resolved.

    Lazy references are used in various places throughout Django, primarily in
    related fields and model signals. Identify those common cases and provide
    more helpful error messages for them.

    The ignore parameter is used by StateApps to exclude swappable models from
    this check.
    """
    pending_models = set(apps._pending_operations) - (ignore or set())
    if not pending_models:
        return []
    from django.db.models import signals
    model_signals = {signal: name for name, signal in vars(signals).items() if isinstance(signal, signals.ModelSignal)}

    def extract_operation(obj):
        """
        Take a callable found in Apps._pending_operations and identify the
        original callable passed to Apps.lazy_model_operation(). If that
        callable was a partial, return the inner, non-partial function and
        any arguments and keyword arguments that were supplied with it.

        obj is a callback defined locally in Apps.lazy_model_operation() and
        annotated there with a `func` attribute so as to imitate a partial.
        """
        operation, args, keywords = (obj, [], {})
        while hasattr(operation, 'func'):
            args.extend(getattr(operation, 'args', []))
            keywords.update(getattr(operation, 'keywords', {}))
            operation = operation.func
        return (operation, args, keywords)

    def app_model_error(model_key):
        try:
            apps.get_app_config(model_key[0])
            model_error = "app '%s' doesn't provide model '%s'" % model_key
        except LookupError:
            model_error = "app '%s' isn't installed" % model_key[0]
        return model_error

    def field_error(model_key, func, args, keywords):
        error_msg = "The field %(field)s was declared with a lazy reference to '%(model)s', but %(model_error)s."
        params = {'model': '.'.join(model_key), 'field': keywords['field'], 'model_error': app_model_error(model_key)}
        return Error(error_msg % params, obj=keywords['field'], id='fields.E307')

    def signal_connect_error(model_key, func, args, keywords):
        error_msg = "%(receiver)s was connected to the '%(signal)s' signal with a lazy reference to the sender '%(model)s', but %(model_error)s."
        receiver = args[0]
        if isinstance(receiver, types.FunctionType):
            description = "The function '%s'" % receiver.__name__
        elif isinstance(receiver, types.MethodType):
            description = "Bound method '%s.%s'" % (receiver.__self__.__class__.__name__, receiver.__name__)
        else:
            description = "An instance of class '%s'" % receiver.__class__.__name__
        signal_name = model_signals.get(func.__self__, 'unknown')
        params = {'model': '.'.join(model_key), 'receiver': description, 'signal': signal_name, 'model_error': app_model_error(model_key)}
        return Error(error_msg % params, obj=receiver.__module__, id='signals.E001')

    def default_error(model_key, func, args, keywords):
        error_msg = '%(op)s contains a lazy reference to %(model)s, but %(model_error)s.'
        params = {'op': func, 'model': '.'.join(model_key), 'model_error': app_model_error(model_key)}
        return Error(error_msg % params, obj=func, id='models.E022')
    known_lazy = {('django.db.models.fields.related', 'resolve_related_class'): field_error, ('django.db.models.fields.related', 'set_managed'): None, ('django.dispatch.dispatcher', 'connect'): signal_connect_error}

    def build_error(model_key, func, args, keywords):
        key = (func.__module__, func.__name__)
        error_fn = known_lazy.get(key, default_error)
        return error_fn(model_key, func, args, keywords) if error_fn else None
    return sorted(filter(None, (build_error(model_key, *extract_operation(func)) for model_key in pending_models for func in apps._pending_operations[model_key])), key=lambda error: error.msg)