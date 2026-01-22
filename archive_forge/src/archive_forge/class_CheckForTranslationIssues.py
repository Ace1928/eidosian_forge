import ast
from hacking import core
import re
class CheckForTranslationIssues(BaseASTChecker):
    name = 'check_for_translation_issues'
    version = '1.0'
    LOGGING_CHECK_DESC = 'K005 Using translated string in logging'
    USING_DEPRECATED_WARN = 'K009 Using the deprecated Logger.warn'
    LOG_MODULES = ('logging', 'oslo_log.log')
    I18N_MODULES = ('keystone.i18n._',)
    TRANS_HELPER_MAP = {'debug': None, 'info': '_LI', 'warning': '_LW', 'error': '_LE', 'exception': '_LE', 'critical': '_LC'}

    def __init__(self, tree, filename):
        super(CheckForTranslationIssues, self).__init__(tree, filename)
        self.logger_names = []
        self.logger_module_names = []
        self.i18n_names = {}
        self.assignments = {}

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        item._parent = node
                        self.visit(item)
            elif isinstance(value, ast.AST):
                value._parent = node
                self.visit(value)

    def _filter_imports(self, module_name, alias):
        """Keep lists of logging and i18n imports."""
        if module_name in self.LOG_MODULES:
            self.logger_module_names.append(alias.asname or alias.name)
        elif module_name in self.I18N_MODULES:
            self.i18n_names[alias.asname or alias.name] = alias.name

    def visit_Import(self, node):
        for alias in node.names:
            self._filter_imports(alias.name, alias)
        return super(CheckForTranslationIssues, self).generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            full_name = '%s.%s' % (node.module, alias.name)
            self._filter_imports(full_name, alias)
        return super(CheckForTranslationIssues, self).generic_visit(node)

    def _find_name(self, node):
        """Return the fully qualified name or a Name or Attribute."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
            method_name = node.attr
            obj_name = self._find_name(node.value)
            if obj_name is None:
                return None
            return obj_name + '.' + method_name
        elif isinstance(node, str):
            return node
        else:
            return None

    def visit_Assign(self, node):
        """Look for 'LOG = logging.getLogger'.

        This handles the simple case:
          name = [logging_module].getLogger(...)

          - or -

          name = [i18n_name](...)

        And some much more comple ones:
          name = [i18n_name](...) % X

          - or -

          self.name = [i18n_name](...) % X

        """
        attr_node_types = (ast.Name, ast.Attribute)
        if len(node.targets) != 1 or not isinstance(node.targets[0], attr_node_types):
            return super(CheckForTranslationIssues, self).generic_visit(node)
        target_name = self._find_name(node.targets[0])
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
            if isinstance(node.value.left, ast.Call) and isinstance(node.value.left.func, ast.Name) and (node.value.left.func.id in self.i18n_names):
                node = ast.Assign(value=node.value.left)
        if not isinstance(node.value, ast.Call):
            self.assignments.pop(target_name, None)
            return super(CheckForTranslationIssues, self).generic_visit(node)
        if isinstance(node.value.func, ast.Name) and node.value.func.id in self.i18n_names:
            self.assignments[target_name] = node.value.func.id
            return super(CheckForTranslationIssues, self).generic_visit(node)
        if not isinstance(node.value.func, ast.Attribute) or not isinstance(node.value.func.value, attr_node_types):
            return super(CheckForTranslationIssues, self).generic_visit(node)
        object_name = self._find_name(node.value.func.value)
        func_name = node.value.func.attr
        if object_name in self.logger_module_names and func_name == 'getLogger':
            self.logger_names.append(target_name)
        return super(CheckForTranslationIssues, self).generic_visit(node)

    def visit_Call(self, node):
        """Look for the 'LOG.*' calls."""
        if isinstance(node.func, ast.Attribute):
            obj_name = self._find_name(node.func.value)
            if isinstance(node.func.value, ast.Name):
                method_name = node.func.attr
            elif isinstance(node.func.value, ast.Attribute):
                obj_name = self._find_name(node.func.value)
                method_name = node.func.attr
            else:
                return super(CheckForTranslationIssues, self).generic_visit(node)
            if obj_name in self.logger_names and method_name == 'warn':
                msg = node.args[0]
                self.add_error(msg, message=self.USING_DEPRECATED_WARN)
            if obj_name not in self.logger_names or method_name not in self.TRANS_HELPER_MAP:
                return super(CheckForTranslationIssues, self).generic_visit(node)
            if not node.args:
                return super(CheckForTranslationIssues, self).generic_visit(node)
            self._process_log_messages(node)
        return super(CheckForTranslationIssues, self).generic_visit(node)

    def _process_log_messages(self, node):
        msg = node.args[0]
        if isinstance(msg, ast.Call) and isinstance(msg.func, ast.Name) and (msg.func.id in self.i18n_names):
            self.add_error(msg, message=self.LOGGING_CHECK_DESC)
        elif isinstance(msg, ast.Name) and msg.id in self.assignments:
            self.add_error(msg, message=self.LOGGING_CHECK_DESC)