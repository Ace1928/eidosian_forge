from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
Generate the frequency values of associated channels.

    Stylesheets:
        - The `axis_label` style is applied.
        - The `annotate` style is partially applied for the font size.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    