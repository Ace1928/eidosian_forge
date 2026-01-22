from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import docutils
from docutils import nodes
from docutils.nodes import Element
class pending_xref_condition(nodes.Inline, nodes.TextElement):
    """Node representing a potential way to create a cross-reference and the
    condition in which this way should be used.

    This node is only allowed to be placed under a :py:class:`pending_xref`
    node.  A **pending_xref** node must contain either no **pending_xref_condition**
    nodes or it must only contains **pending_xref_condition** nodes.

    The cross-reference resolver will replace a :py:class:`pending_xref` which
    contains **pending_xref_condition** nodes by the content of exactly one of
    those **pending_xref_condition** nodes' content. It uses the **condition**
    attribute to decide which **pending_xref_condition** node's content to
    use. For example, let us consider how the cross-reference resolver acts on::

        <pending_xref refdomain="py" reftarget="io.StringIO ...>
            <pending_xref_condition condition="resolved">
                <literal>
                    StringIO
            <pending_xref_condition condition="*">
                <literal>
                    io.StringIO

    If the cross-reference resolver successfully resolves the cross-reference,
    then it rewrites the **pending_xref** as::

        <reference>
            <literal>
                StringIO

    Otherwise, if the cross-reference resolution failed, it rewrites the
    **pending_xref** as::

        <reference>
            <literal>
                io.StringIO

    The **pending_xref_condition** node should have **condition** attribute.
    Domains can be store their individual conditions into the attribute to
    filter contents on resolving phase.  As a reserved condition name,
    ``condition="*"`` is used for the fallback of resolution failure.
    Additionally, as a recommended condition name, ``condition="resolved"``
    represents a resolution success in the intersphinx module.

    .. versionadded:: 4.0
    """