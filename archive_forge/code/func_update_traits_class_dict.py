import abc
import copy as copy_module
import inspect
import os
import pickle
import re
import types
import warnings
import weakref
from types import FunctionType
from . import __version__ as TraitsVersion
from .adaptation.adaptation_error import AdaptationError
from .constants import DefaultValue, TraitKind
from .ctrait import CTrait, __newobj__
from .ctraits import CHasTraits
from .observation import api as observe_api
from .traits import (
from .trait_types import Any, Bool, Disallow, Event, Python, Str
from .trait_notifiers import (
from .trait_base import (
from .trait_errors import TraitError
from .util.deprecated import deprecated
from .util._traitsui_helpers import check_traitsui_major_version
from .trait_converters import check_trait, mapped_trait_for, trait_for
def update_traits_class_dict(class_name, bases, class_dict):
    """ Processes all of the traits related data in the class dictionary.

    This is called during the construction of a new HasTraits class. The first
    three parameters have the same interpretation as the corresponding
    parameters of ``type.__new__``. This function modifies ``class_dict``
    in-place.

    Parameters
    ----------
    class_name : str
        The name of the HasTraits class.
    bases : tuple
        The base classes for the HasTraits class.
    class_dict : dict
        A dictionary of class members.
    """
    base_traits = {}
    class_traits = {}
    prefix_traits = {}
    listeners = {}
    prefix_list = []
    view_elements = {}
    observers = {}
    hastraits_bases = [base for base in bases if base.__dict__.get(ClassTraits) is not None]
    inherited_class_traits = [base.__dict__.get(ClassTraits) for base in hastraits_bases]
    for name, value in list(class_dict.items()):
        value = check_trait(value)
        rc = isinstance(value, CTrait)
        if not rc and isinstance(value, ForwardProperty):
            rc = True
            getter = _property_method(class_dict, '_get_' + name)
            setter = _property_method(class_dict, '_set_' + name)
            if setter is None and getter is not None:
                if getattr(getter, 'settable', False):
                    setter = HasTraits._set_traits_cache
                elif getattr(getter, 'flushable', False):
                    setter = HasTraits._flush_traits_cache
            validate = _property_method(class_dict, '_validate_' + name)
            if validate is None:
                validate = value.validate
            value = Property(getter, setter, validate, True, value.handler, **value.metadata)
        if rc:
            del class_dict[name]
            if name[-1:] != '_':
                base_traits[name] = class_traits[name] = value
                value_type = value.type
                if value_type == 'trait':
                    handler = value.handler
                    if handler is not None:
                        if handler.has_items:
                            items_trait = _clone_trait(handler.items_event(), value.__dict__)
                            if items_trait.instance_handler == '_list_changed_handler':
                                items_trait.instance_handler = '_list_items_changed_handler'
                            class_traits[name + '_items'] = items_trait
                        if handler.is_mapped:
                            class_traits[name + '_'] = mapped_trait_for(value, name)
                elif value_type == 'delegate':
                    if value._listenable is not False:
                        listeners[name] = ('delegate', get_delegate_pattern(name, value))
                elif value_type == 'event':
                    on_trait_change = value.on_trait_change
                    if isinstance(on_trait_change, str):
                        listeners[name] = ('event', on_trait_change)
            else:
                name = name[:-1]
                prefix_list.append(name)
                prefix_traits[name] = value
        elif is_unbound_method_type(value):
            pattern = getattr(value, 'on_trait_change', None)
            if pattern is not None:
                listeners[name] = ('method', pattern)
            observer_states = getattr(value, '_observe_inputs', None)
            if observer_states is not None:
                observers[name] = observer_states
        elif isinstance(value, property):
            class_traits[name] = generic_trait
        elif isinstance(value, AbstractViewElement):
            view_elements[name] = value
            del class_dict[name]
        else:
            for ct in inherited_class_traits:
                if name in ct:
                    ictrait = ct[name]
                    if ictrait.type in CantHaveDefaultValue:
                        raise TraitError("Cannot specify a default value for the %s trait '%s'. You must override the the trait definition instead." % (ictrait.type, name))
                    default_value = value
                    class_traits[name] = value = ictrait(default_value)
                    if value.setattr_original_value:
                        value.set_default_value(DefaultValue.missing, default_value)
                    else:
                        value.set_default_value(DefaultValue.missing, value.default)
                    del class_dict[name]
                    break
    migrated_properties = {}
    for base in hastraits_bases:
        base_dict = base.__dict__
        for name, value in base_dict.get(ListenerTraits).items():
            if name not in class_traits and name not in class_dict:
                listeners[name] = value
        for name, states in base_dict[ObserverTraits].items():
            if name not in class_traits and name not in class_dict:
                observers[name] = states
        for name, value in base_dict.get(BaseTraits).items():
            if name not in base_traits:
                property_info = value.property_fields
                if property_info is not None:
                    key = id(value)
                    migrated_properties[key] = value = migrate_property(name, value, property_info, class_dict)
                base_traits[name] = value
        for name, value in base_dict.get(ClassTraits).items():
            if name not in class_traits:
                property_info = value.property_fields
                if property_info is not None:
                    new_value = migrated_properties.get(id(value))
                    if new_value is not None:
                        value = new_value
                    else:
                        value = migrate_property(name, value, property_info, class_dict)
                class_traits[name] = value
        base_prefix_traits = base_dict.get(PrefixTraits)
        for name in base_prefix_traits['*']:
            if name not in prefix_list:
                prefix_list.append(name)
                prefix_traits[name] = base_prefix_traits[name]
    if prefix_traits.get('') is None:
        prefix_list.append('')
        prefix_traits[''] = Python().as_ctrait()
    prefix_traits['*'] = prefix_list
    prefix_list.sort(key=len, reverse=True)
    instance_traits = _get_instance_handlers(class_dict, hastraits_bases)
    anytrait = _get_def(class_name, class_dict, bases, '_anytrait_changed')
    if anytrait is not None:
        anytrait = StaticAnytraitChangeNotifyWrapper(anytrait)
        prefix_traits['@'] = anytrait
    cloned = set()
    for name in list(class_traits.keys()):
        trait = class_traits[name]
        handlers = [anytrait, _get_def(class_name, class_dict, bases, '_%s_changed' % name), _get_def(class_name, class_dict, bases, '_%s_fired' % name)]
        instance_handler = trait.instance_handler
        if instance_handler is not None and name in instance_traits or (instance_handler == '_list_items_changed_handler' and name[-6:] == '_items' and (name[:-6] in instance_traits)):
            handlers.append(getattr(HasTraits, instance_handler))
        events = trait.event
        if events is not None:
            if isinstance(events, str):
                events = [events]
            for event in events:
                handlers.append(_get_def(class_name, class_dict, bases, '_%s_changed' % event))
                handlers.append(_get_def(class_name, class_dict, bases, '_%s_fired' % event))
        handlers = [h for h in handlers if h is not None]
        default = _get_def(class_name, class_dict, [], '_%s_default' % name)
        if len(handlers) > 0 or default is not None:
            if name not in cloned:
                cloned.add(name)
                class_traits[name] = trait = _clone_trait(trait)
            if len(handlers) > 0:
                _add_notifiers(trait._notifiers(True), handlers)
            if default is not None:
                trait.set_default_value(DefaultValue.callable, default)
        if trait.type == 'property' and trait.depends_on is not None:
            cached = trait.cached
            if cached is True:
                cached = TraitsCache + name
            depends_on = trait.depends_on
            if isinstance(depends_on, SequenceTypes):
                depends_on = ','.join(depends_on)
            else:
                depends_on = ' ' + depends_on
            listeners[name] = ('property', cached, depends_on)
        if trait.type == 'property' and trait.observe is not None:
            observer_state = _create_property_observe_state(observe=trait.observe, property_name=name, cached=trait.cached)
            observers[name] = [observer_state]
    class_dict[BaseTraits] = base_traits
    class_dict[ClassTraits] = class_traits
    class_dict[InstanceTraits] = instance_traits
    class_dict[PrefixTraits] = prefix_traits
    class_dict[ListenerTraits] = listeners
    class_dict[ObserverTraits] = observers
    class_dict[ViewTraits] = view_elements