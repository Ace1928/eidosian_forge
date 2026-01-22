from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.IFrame_pb2 import IFrame as IFrameProto
from streamlit.runtime.metrics_util import gather_metrics
class IframeMixin:

    @gather_metrics('_iframe')
    def _iframe(self, src: str, width: int | None=None, height: int | None=None, scrolling: bool=False) -> DeltaGenerator:
        """Load a remote URL in an iframe.

        Parameters
        ----------
        src : str
            The URL of the page to embed.
        width : int
            The width of the frame in CSS pixels. Defaults to the app's
            default element width.
        height : int
            The height of the frame in CSS pixels. Defaults to 150.
        scrolling : bool
            If True, show a scrollbar when the content is larger than the iframe.
            Otherwise, do not show a scrollbar. Defaults to False.

        """
        iframe_proto = IFrameProto()
        marshall(iframe_proto, src=src, width=width, height=height, scrolling=scrolling)
        return self.dg._enqueue('iframe', iframe_proto)

    @gather_metrics('_html')
    def _html(self, html: str, width: int | None=None, height: int | None=None, scrolling: bool=False) -> DeltaGenerator:
        """Display an HTML string in an iframe.

        Parameters
        ----------
        html : str
            The HTML string to embed in the iframe.
        width : int
            The width of the frame in CSS pixels. Defaults to the app's
            default element width.
        height : int
            The height of the frame in CSS pixels. Defaults to 150.
        scrolling : bool
            If True, show a scrollbar when the content is larger than the iframe.
            Otherwise, do not show a scrollbar. Defaults to False.

        """
        iframe_proto = IFrameProto()
        marshall(iframe_proto, srcdoc=html, width=width, height=height, scrolling=scrolling)
        return self.dg._enqueue('iframe', iframe_proto)

    @property
    def dg(self) -> DeltaGenerator:
        """Get our DeltaGenerator."""
        return cast('DeltaGenerator', self)