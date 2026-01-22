import jinja2
from minerl.herobraine.hero.handler import Handler
class ServerQuitWhenAnyAgentFinishes(Handler):
    """ Forces the server to quit if any of the agents involved quits.
    Has no parameters."""

    def to_string(self) -> str:
        return 'server_quit_when_any_agent_finishes'

    def xml_template(self) -> str:
        return str('<ServerQuitWhenAnyAgentFinishes/>\n            ')