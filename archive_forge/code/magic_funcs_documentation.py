from __future__ import annotations
from typing import Any
from streamlit.runtime.metrics_util import gather_metrics
The function that gets magic-ified into Streamlit apps.
    This is just st.write, but returns the arguments you passed to it.
    