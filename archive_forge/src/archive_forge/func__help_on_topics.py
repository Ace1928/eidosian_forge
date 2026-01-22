import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def _help_on_topics(dummy):
    """Write out the help for topics to outfile"""
    topics = topic_registry.keys()
    lmax = max((len(topic) for topic in topics))
    out = []
    for topic in topics:
        summary = topic_registry.get_summary(topic)
        out.append('%-*s %s\n' % (lmax, topic, summary))
    return ''.join(out)