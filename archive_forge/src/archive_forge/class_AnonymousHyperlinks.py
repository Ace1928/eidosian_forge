import sys
import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class AnonymousHyperlinks(Transform):
    """
    Link anonymous references to targets.  Given::

        <paragraph>
            <reference anonymous="1">
                internal
            <reference anonymous="1">
                external
        <target anonymous="1" ids="id1">
        <target anonymous="1" ids="id2" refuri="http://external">

    Corresponding references are linked via "refid" or resolved via "refuri"::

        <paragraph>
            <reference anonymous="1" refid="id1">
                text
            <reference anonymous="1" refuri="http://external">
                external
        <target anonymous="1" ids="id1">
        <target anonymous="1" ids="id2" refuri="http://external">
    """
    default_priority = 440

    def apply(self):
        anonymous_refs = []
        anonymous_targets = []
        for node in self.document.traverse(nodes.reference):
            if node.get('anonymous'):
                anonymous_refs.append(node)
        for node in self.document.traverse(nodes.target):
            if node.get('anonymous'):
                anonymous_targets.append(node)
        if len(anonymous_refs) != len(anonymous_targets):
            msg = self.document.reporter.error('Anonymous hyperlink mismatch: %s references but %s targets.\nSee "backrefs" attribute for IDs.' % (len(anonymous_refs), len(anonymous_targets)))
            msgid = self.document.set_id(msg)
            for ref in anonymous_refs:
                prb = nodes.problematic(ref.rawsource, ref.rawsource, refid=msgid)
                prbid = self.document.set_id(prb)
                msg.add_backref(prbid)
                ref.replace_self(prb)
            return
        for ref, target in zip(anonymous_refs, anonymous_targets):
            target.referenced = 1
            while True:
                if target.hasattr('refuri'):
                    ref['refuri'] = target['refuri']
                    ref.resolved = 1
                    break
                else:
                    if not target['ids']:
                        target = self.document.ids[target['refid']]
                        continue
                    ref['refid'] = target['ids'][0]
                    self.document.note_refid(ref)
                    break