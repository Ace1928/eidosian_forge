from inspect import signature
from docutils import nodes
from sphinx.domains.python import PyFunction
from sphinx.ext.autodoc import FunctionDocumenter
from celery.app.task import BaseTask
class TaskDirective(PyFunction):
    """Sphinx task directive."""

    def get_signature_prefix(self, sig):
        return [nodes.Text(self.env.config.celery_task_prefix)]