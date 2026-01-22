import sys
from os import environ
from os.path import join
from copy import copy
from types import CodeType
from functools import partial
from kivy.factory import Factory
from kivy.lang.parser import (
from kivy.logger import Logger
from kivy.utils import QueryDict
from kivy.cache import Cache
from kivy import kivy_data_dir
from kivy.context import register_context
from kivy.resources import resource_find
from kivy._event import Observable, EventDispatcher
def _apply_rule(self, widget, rule, rootrule, template_ctx=None, ignored_consts=set(), rule_children=None):
    assert rule not in self.rulectx
    self.rulectx[rule] = rctx = {'ids': {'root': widget.proxy_ref}, 'set': [], 'hdl': []}
    assert rootrule in self.rulectx
    rctx = self.rulectx[rootrule]
    if template_ctx is not None:
        rctx['ids']['ctx'] = QueryDict(template_ctx)
    if rule.id:
        rule.id = rule.id.split('#', 1)[0].strip()
        rctx['ids'][rule.id] = widget.proxy_ref
        _ids = dict(rctx['ids'])
        _root = _ids.pop('root')
        _new_ids = _root.ids
        for _key, _value in _ids.items():
            if _value == _root:
                continue
            _new_ids[_key] = _value
        _root.ids = _new_ids
    rule.create_missing(widget)
    if rule.canvas_before:
        with widget.canvas.before:
            self._build_canvas(widget.canvas.before, widget, rule.canvas_before, rootrule)
    if rule.canvas_root:
        with widget.canvas:
            self._build_canvas(widget.canvas, widget, rule.canvas_root, rootrule)
    if rule.canvas_after:
        with widget.canvas.after:
            self._build_canvas(widget.canvas.after, widget, rule.canvas_after, rootrule)
    Factory_get = Factory.get
    Factory_is_template = Factory.is_template
    for crule in rule.children:
        cname = crule.name
        if cname in ('canvas', 'canvas.before', 'canvas.after'):
            raise ParserException(crule.ctx, crule.line, 'Canvas instructions added in kv must be declared before child widgets.')
        cls = Factory_get(cname)
        if Factory_is_template(cname):
            ctx = {}
            idmap = copy(global_idmap)
            idmap.update({'root': rctx['ids']['root']})
            if 'ctx' in rctx['ids']:
                idmap.update({'ctx': rctx['ids']['ctx']})
            try:
                for prule in crule.properties.values():
                    value = prule.co_value
                    if type(value) is CodeType:
                        value = eval(value, idmap)
                    ctx[prule.name] = value
                for prule in crule.handlers:
                    value = eval(prule.value, idmap)
                    ctx[prule.name] = value
            except Exception as e:
                tb = sys.exc_info()[2]
                raise BuilderException(prule.ctx, prule.line, '{}: {}'.format(e.__class__.__name__, e), cause=tb)
            child = cls(**ctx)
            widget.add_widget(child)
            if crule.id:
                rctx['ids'][crule.id] = child
        else:
            child = cls(__no_builder=True)
            widget.add_widget(child)
            child.apply_class_lang_rules(root=rctx['ids']['root'], rule_children=rule_children)
            self._apply_rule(child, crule, rootrule, rule_children=rule_children)
            if rule_children is not None:
                rule_children.append(child)
    if rule.properties:
        rctx['set'].append((widget.proxy_ref, list(rule.properties.values())))
        for key, crule in rule.properties.items():
            if crule.ignore_prev:
                Builder.unbind_property(widget, key)
    if rule.handlers:
        rctx['hdl'].append((widget.proxy_ref, rule.handlers))
    if rootrule is not rule:
        del self.rulectx[rule]
        return
    try:
        rule = None
        for widget_set, rules in reversed(rctx['set']):
            for rule in rules:
                assert isinstance(rule, ParserRuleProperty)
                key = rule.name
                value = rule.co_value
                if type(value) is CodeType:
                    value, bound = create_handler(widget_set, widget_set, key, value, rule, rctx['ids'])
                    if widget_set != widget or bound or key not in ignored_consts:
                        setattr(widget_set, key, value)
                elif widget_set != widget or key not in ignored_consts:
                    setattr(widget_set, key, value)
    except Exception as e:
        if rule is not None:
            tb = sys.exc_info()[2]
            raise BuilderException(rule.ctx, rule.line, '{}: {}'.format(e.__class__.__name__, e), cause=tb)
        raise e
    try:
        crule = None
        for widget_set, rules in rctx['hdl']:
            for crule in rules:
                assert isinstance(crule, ParserRuleProperty)
                assert crule.name.startswith('on_')
                key = crule.name
                if not widget_set.is_event_type(key):
                    key = key[3:]
                idmap = copy(global_idmap)
                idmap.update(rctx['ids'])
                idmap['self'] = widget_set.proxy_ref
                if not widget_set.fbind(key, custom_callback, crule, idmap):
                    raise AttributeError(key)
                if crule.name == 'on_parent':
                    Factory.Widget.parent.dispatch(widget_set.__self__)
    except Exception as e:
        if crule is not None:
            tb = sys.exc_info()[2]
            raise BuilderException(crule.ctx, crule.line, '{}: {}'.format(e.__class__.__name__, e), cause=tb)
        raise e
    del self.rulectx[rootrule]