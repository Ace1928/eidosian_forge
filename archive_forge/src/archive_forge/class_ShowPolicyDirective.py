import os
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from oslo_config import cfg
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from oslo_policy import generator
class ShowPolicyDirective(rst.Directive):
    has_content = False
    option_spec = {'config-file': directives.unchanged}

    def run(self):
        env = self.state.document.settings.env
        app = env.app
        config_file = self.options.get('config-file')
        if not config_file and hasattr(env.config, 'policy_generator_config_file'):
            config_file = env.config.policy_generator_config_file
        candidates = [config_file, os.path.join(app.srcdir, config_file)]
        for c in candidates:
            if os.path.isfile(c):
                config_path = c
                break
        else:
            raise ValueError('could not find config file in: %s' % str(candidates))
        self.info('loading config file %s' % config_path)
        conf = cfg.ConfigOpts()
        opts = generator.GENERATOR_OPTS + generator.RULE_OPTS
        conf.register_cli_opts(opts)
        conf.register_opts(opts)
        conf(args=['--config-file', config_path])
        namespaces = conf.namespace[:]
        result = statemachine.ViewList()
        source_name = '<' + __name__ + '>'
        for line in _format_policy(namespaces):
            result.append(line, source_name)
        node = nodes.section()
        node.document = self.state.document
        with logging.skip_warningiserror():
            nested_parse_with_titles(self.state, result, node)
        return node.children