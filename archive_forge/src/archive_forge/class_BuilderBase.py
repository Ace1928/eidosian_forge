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
class BuilderBase(object):
    """The Builder is responsible for creating a :class:`Parser` for parsing a
    kv file, merging the results into its internal rules, templates, etc.

    By default, :class:`Builder` is a global Kivy instance used in widgets
    that you can use to load other kv files in addition to the default ones.
    """

    def __init__(self):
        super(BuilderBase, self).__init__()
        self._match_cache = {}
        self._match_name_cache = {}
        self.files = []
        self.dynamic_classes = {}
        self.templates = {}
        self.rules = []
        self.rulectx = {}

    @classmethod
    def create_from(cls, builder):
        """Creates a instance of the class, and initializes to the state of
        ``builder``.

        :param builder: The builder to initialize from.
        :return: A new instance of this class.
        """
        obj = cls()
        obj._match_cache = copy(builder._match_cache)
        obj._match_name_cache = copy(builder._match_name_cache)
        obj.files = copy(builder.files)
        obj.dynamic_classes = copy(builder.dynamic_classes)
        obj.templates = copy(builder.templates)
        obj.rules = list(builder.rules)
        assert not builder.rulectx
        obj.rulectx = dict(builder.rulectx)
        return obj

    def load_file(self, filename, encoding='utf8', **kwargs):
        """Insert a file into the language builder and return the root widget
        (if defined) of the kv file.

        :parameters:
            `rulesonly`: bool, defaults to False
                If True, the Builder will raise an exception if you have a root
                widget inside the definition.

            `encoding`: File character encoding. Defaults to utf-8,
        """
        filename = resource_find(filename) or filename
        if __debug__:
            trace('Lang: load file %s, using %s encoding', filename, encoding)
        kwargs['filename'] = filename
        with open(filename, 'r', encoding=encoding) as fd:
            data = fd.read()
            return self.load_string(data, **kwargs)

    def unload_file(self, filename):
        """Unload all rules associated with a previously imported file.

        .. versionadded:: 1.0.8

        .. warning::

            This will not remove rules or templates already applied/used on
            current widgets. It will only effect the next widgets creation or
            template invocation.
        """
        filename = resource_find(filename) or filename
        self.rules = [x for x in self.rules if x[1].ctx.filename != filename]
        self._clear_matchcache()
        templates = {}
        for x, y in self.templates.items():
            if y[2] != filename:
                templates[x] = y
        self.templates = templates
        if filename in self.files:
            self.files.remove(filename)
        Factory.unregister_from_filename(filename)

    def load_string(self, string, **kwargs):
        '''Insert a string into the Language Builder and return the root widget
        (if defined) of the kv string.

        :Parameters:
            `rulesonly`: bool, defaults to False
                If True, the Builder will raise an exception if you have a root
                widget inside the definition.
            `filename`: str, defaults to None
                If specified, the filename used to index the kv rules.

        The filename parameter can be used to unload kv strings in the same way
        as you unload kv files. This can be achieved using pseudo file names
        e.g.::

            Build.load_string("""
                <MyRule>:
                    Label:
                        text="Hello"
            """, filename="myrule.kv")

        can be unloaded via::

            Build.unload_file("myrule.kv")

        '''
        kwargs.setdefault('rulesonly', False)
        self._current_filename = fn = kwargs.get('filename', None)
        if fn in self.files:
            Logger.warning('Lang: The file {} is loaded multiples times, you might have unwanted behaviors.'.format(fn))
        try:
            parser = Parser(content=string, filename=fn)
            self.rules.extend(parser.rules)
            self._clear_matchcache()
            for name, cls, template in parser.templates:
                self.templates[name] = (cls, template, fn)
                Factory.register(name, cls=partial(self.template, name), is_template=True, warn=True)
            for name, baseclasses in parser.dynamic_classes.items():
                Factory.register(name, baseclasses=baseclasses, filename=fn, warn=True)
            if kwargs['rulesonly'] and parser.root:
                filename = kwargs.get('rulesonly', '<string>')
                raise Exception('The file <%s> contain also non-rules directives' % filename)
            if fn and (parser.templates or parser.dynamic_classes or parser.rules):
                self.files.append(fn)
            if parser.root:
                widget = Factory.get(parser.root.name)(__no_builder=True)
                rule_children = []
                widget.apply_class_lang_rules(root=widget, rule_children=rule_children)
                self._apply_rule(widget, parser.root, parser.root, rule_children=rule_children)
                for child in rule_children:
                    child.dispatch('on_kv_post', widget)
                widget.dispatch('on_kv_post', widget)
                return widget
        finally:
            self._current_filename = None

    def template(self, *args, **ctx):
        """Create a specialized template using a specific context.

        .. versionadded:: 1.0.5

        With templates, you can construct custom widgets from a kv lang
        definition by giving them a context. Check :ref:`Template usage
        <template_usage>`.
        """
        name = args[0]
        if name not in self.templates:
            raise Exception('Unknown <%s> template name' % name)
        baseclasses, rule, fn = self.templates[name]
        key = '%s|%s' % (name, baseclasses)
        cls = Cache.get('kv.lang', key)
        if cls is None:
            rootwidgets = []
            for basecls in baseclasses.split('+'):
                rootwidgets.append(Factory.get(basecls))
            cls = type(name, tuple(rootwidgets), {})
            Cache.append('kv.lang', key, cls)
        widget = cls()
        proxy_ctx = {k: get_proxy(v) for k, v in ctx.items()}
        self._apply_rule(widget, rule, rule, template_ctx=proxy_ctx)
        return widget

    def apply_rules(self, widget, rule_name, ignored_consts=set(), rule_children=None, dispatch_kv_post=False):
        """Search all the rules that match the name `rule_name`
        and apply them to `widget`.

        .. versionadded:: 1.10.0

        :Parameters:

            `widget`: :class:`~kivy.uix.widget.Widget`
                The widget to whom the matching rules should be applied to.
            `ignored_consts`: set
                A set or list type whose elements are property names for which
                constant KV rules (i.e. those that don't create bindings) of
                that widget will not be applied. This allows e.g. skipping
                constant rules that overwrite a value initialized in python.
            `rule_children`: list
                If not ``None``, it should be a list that will be populated
                with all the widgets created by the kv rules being applied.

                .. versionchanged:: 1.11.0

            `dispatch_kv_post`: bool
                Normally the class `Widget` dispatches the `on_kv_post` event
                to widgets created during kv rule application.
                But if the rules are manually applied by calling :meth:`apply`,
                that may not happen, so if this is `True`, we will dispatch the
                `on_kv_post` event where needed after applying the rules to
                `widget` (we won't dispatch it for `widget` itself).

                Defaults to False.

                .. versionchanged:: 1.11.0
        """
        rules = self.match_rule_name(rule_name)
        if __debug__:
            trace('Lang: Found %d rules for %s' % (len(rules), rule_name))
        if not rules:
            return
        if dispatch_kv_post:
            rule_children = rule_children if rule_children is not None else []
        for rule in rules:
            self._apply_rule(widget, rule, rule, ignored_consts=ignored_consts, rule_children=rule_children)
        if dispatch_kv_post:
            for w in rule_children:
                w.dispatch('on_kv_post', widget)

    def apply(self, widget, ignored_consts=set(), rule_children=None, dispatch_kv_post=False):
        """Search all the rules that match the widget and apply them.

        :Parameters:

            `widget`: :class:`~kivy.uix.widget.Widget`
                The widget whose class rules should be applied to this widget.
            `ignored_consts`: set
                A set or list type whose elements are property names for which
                constant KV rules (i.e. those that don't create bindings) of
                that widget will not be applied. This allows e.g. skipping
                constant rules that overwrite a value initialized in python.
            `rule_children`: list
                If not ``None``, it should be a list that will be populated
                with all the widgets created by the kv rules being applied.

                .. versionchanged:: 1.11.0

            `dispatch_kv_post`: bool
                Normally the class `Widget` dispatches the `on_kv_post` event
                to widgets created during kv rule application.
                But if the rules are manually applied by calling :meth:`apply`,
                that may not happen, so if this is `True`, we will dispatch the
                `on_kv_post` event where needed after applying the rules to
                `widget` (we won't dispatch it for `widget` itself).

                Defaults to False.

                .. versionchanged:: 1.11.0
        """
        rules = self.match(widget)
        if __debug__:
            trace('Lang: Found %d rules for %s' % (len(rules), widget))
        if not rules:
            return
        if dispatch_kv_post:
            rule_children = rule_children if rule_children is not None else []
        for rule in rules:
            self._apply_rule(widget, rule, rule, ignored_consts=ignored_consts, rule_children=rule_children)
        if dispatch_kv_post:
            for w in rule_children:
                w.dispatch('on_kv_post', widget)

    def _clear_matchcache(self):
        self._match_cache.clear()
        self._match_name_cache.clear()

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

    def match(self, widget):
        """Return a list of :class:`ParserRule` objects matching the widget.
        """
        cache = self._match_cache
        k = (widget.__class__, tuple(widget.cls))
        if k in cache:
            return cache[k]
        rules = []
        for selector, rule in self.rules:
            if selector.match(widget):
                if rule.avoid_previous_rules:
                    del rules[:]
                rules.append(rule)
        cache[k] = rules
        return rules

    def match_rule_name(self, rule_name):
        """Return a list of :class:`ParserRule` objects matching the widget.
        """
        cache = self._match_name_cache
        rule_name = str(rule_name)
        k = rule_name.lower()
        if k in cache:
            return cache[k]
        rules = []
        for selector, rule in self.rules:
            if selector.match_rule_name(rule_name):
                if rule.avoid_previous_rules:
                    del rules[:]
                rules.append(rule)
        cache[k] = rules
        return rules

    def sync(self):
        """Execute all the waiting operations, such as the execution of all the
        expressions related to the canvas.

        .. versionadded:: 1.7.0
        """
        global _delayed_start
        next_args = _delayed_start
        if next_args is None:
            return
        while next_args is not StopIteration:
            try:
                call_fn(next_args[:-1], None, None)
            except ReferenceError:
                pass
            args = next_args
            next_args = args[-1]
            args[-1] = None
        _delayed_start = None

    def unbind_widget(self, uid):
        """Unbind all the handlers created by the KV rules of the
        widget. The :attr:`kivy.uix.widget.Widget.uid` is passed here
        instead of the widget itself, because Builder is using it in the
        widget destructor.

        This effectively clears all the KV rules associated with this widget.
        For example:

        .. code-block:: python

            >>> w = Builder.load_string('''
            ... Widget:
            ...     height: self.width / 2. if self.disabled else self.width
            ...     x: self.y + 50
            ... ''')
            >>> w.size
            [100, 100]
            >>> w.pos
            [50, 0]
            >>> w.width = 500
            >>> w.size
            [500, 500]
            >>> Builder.unbind_widget(w.uid)
            >>> w.width = 222
            >>> w.y = 500
            >>> w.size
            [222, 500]
            >>> w.pos
            [50, 500]

        .. versionadded:: 1.7.2
        """
        if uid not in _handlers:
            return
        for prop_callbacks in _handlers[uid].values():
            for callbacks in prop_callbacks:
                for f, k, fn, bound_uid in callbacks:
                    if fn is None:
                        continue
                    try:
                        f.unbind_uid(k, bound_uid)
                    except ReferenceError:
                        pass
        del _handlers[uid]

    def unbind_property(self, widget, name):
        """Unbind the handlers created by all the rules of the widget that set
        the name.

        This effectively clears all the rules of widget that take the form::

            name: rule

        For example:

        .. code-block:: python

            >>> w = Builder.load_string('''
            ... Widget:
            ...     height: self.width / 2. if self.disabled else self.width
            ...     x: self.y + 50
            ... ''')
            >>> w.size
            [100, 100]
            >>> w.pos
            [50, 0]
            >>> w.width = 500
            >>> w.size
            [500, 500]
            >>> Builder.unbind_property(w, 'height')
            >>> w.width = 222
            >>> w.size
            [222, 500]
            >>> w.y = 500
            >>> w.pos
            [550, 500]

        .. versionadded:: 1.9.1
        """
        uid = widget.uid
        if uid not in _handlers:
            return
        prop_handlers = _handlers[uid]
        if name not in prop_handlers:
            return
        for callbacks in prop_handlers[name]:
            for f, k, fn, bound_uid in callbacks:
                if fn is None:
                    continue
                try:
                    f.unbind_uid(k, bound_uid)
                except ReferenceError:
                    pass
        del prop_handlers[name]
        if not prop_handlers:
            del _handlers[uid]

    def _build_canvas(self, canvas, widget, rule, rootrule):
        global Instruction
        if Instruction is None:
            Instruction = Factory.get('Instruction')
        idmap = copy(self.rulectx[rootrule]['ids'])
        for crule in rule.children:
            name = crule.name
            if name == 'Clear':
                canvas.clear()
                continue
            instr = Factory.get(name)()
            if not isinstance(instr, Instruction):
                raise BuilderException(crule.ctx, crule.line, 'You can add only graphics Instruction in canvas.')
            try:
                for prule in crule.properties.values():
                    key = prule.name
                    value = prule.co_value
                    if type(value) is CodeType:
                        value, _ = create_handler(widget, instr.proxy_ref, key, value, prule, idmap, True)
                    setattr(instr, key, value)
            except Exception as e:
                tb = sys.exc_info()[2]
                raise BuilderException(prule.ctx, prule.line, '{}: {}'.format(e.__class__.__name__, e), cause=tb)