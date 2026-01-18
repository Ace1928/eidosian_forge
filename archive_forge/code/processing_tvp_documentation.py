from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

        Compute the time of the video.

        Args:
            logits (`torch.Tensor`):
                The logits output of TvpForVideoGrounding.
            video_durations (`float`):
                The video's duration.

        Returns:
            start (`float`):
                The start time of the video.
            end (`float`):
                The end time of the video.
        