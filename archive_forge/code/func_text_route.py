import textwrap
import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata
@wrappers.Request.application
def text_route(self, request):
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    tag = request.args.get('tag')
    markdown_arg = request.args.get('markdown')
    enable_markdown = markdown_arg != 'false'
    response = self.text_impl(ctx, run, tag, experiment, enable_markdown)
    return http_util.Respond(request, response, 'application/json')