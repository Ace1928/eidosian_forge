from typing import List, Optional
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import to_numpy

        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.

        Example:
        ```python
        >>> from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

        >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> processor = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody")
        >>> unconditional_inputs = processor.get_unconditional_inputs(num_samples=1)

        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```