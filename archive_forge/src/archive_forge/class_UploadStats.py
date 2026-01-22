import contextlib
from datetime import datetime
import sys
import time
class UploadStats:
    """Statistics of uploading."""

    def __init__(self):
        self._last_summarized_timestamp = time.time()
        self._last_data_added_timestamp = 0
        self._num_scalars = 0
        self._num_tensors = 0
        self._num_tensors_skipped = 0
        self._tensor_bytes = 0
        self._tensor_bytes_skipped = 0
        self._num_blobs = 0
        self._num_blobs_skipped = 0
        self._blob_bytes = 0
        self._blob_bytes_skipped = 0
        self._plugin_names = set()

    def add_scalars(self, num_scalars):
        """Add a batch of scalars.

        Args:
          num_scalars: Number of scalars uploaded in this batch.
        """
        self._refresh_last_data_added_timestamp()
        self._num_scalars += num_scalars

    def add_tensors(self, num_tensors, num_tensors_skipped, tensor_bytes, tensor_bytes_skipped):
        """Add a batch of tensors.

        Args:
          num_tensors: Number of tensors encountered in this batch, including
            the ones skipped due to reasons such as large exceeding limit.
          num_tensors: Number of tensors skipped. This describes a subset of
            `num_tensors` and hence must be `<= num_tensors`.
          tensor_bytes: Total byte size of tensors encountered in this batch,
            including the skipped ones.
          tensor_bytes_skipped: Total byte size of the tensors skipped due to
            reasons such as size exceeding limit.
        """
        assert num_tensors_skipped <= num_tensors
        assert tensor_bytes_skipped <= tensor_bytes
        self._refresh_last_data_added_timestamp()
        self._num_tensors += num_tensors
        self._num_tensors_skipped += num_tensors_skipped
        self._tensor_bytes += tensor_bytes
        self._tensor_bytes_skipped = tensor_bytes_skipped

    def add_blob(self, blob_bytes, is_skipped):
        """Add a blob.

        Args:
          blob_bytes: Byte size of the blob.
          is_skipped: Whether the uploading of the blob is skipped due to
            reasons such as size exceeding limit.
        """
        self._refresh_last_data_added_timestamp()
        self._num_blobs += 1
        self._blob_bytes += blob_bytes
        if is_skipped:
            self._num_blobs_skipped += 1
            self._blob_bytes_skipped += blob_bytes

    def add_plugin(self, plugin_name):
        """Add a plugin.

        Args:
          plugin_name: Name of the plugin.
        """
        self._refresh_last_data_added_timestamp()
        self._plugin_names.add(plugin_name)

    @property
    def num_scalars(self):
        return self._num_scalars

    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def num_tensors_skipped(self):
        return self._num_tensors_skipped

    @property
    def tensor_bytes(self):
        return self._tensor_bytes

    @property
    def tensor_bytes_skipped(self):
        return self._tensor_bytes_skipped

    @property
    def num_blobs(self):
        return self._num_blobs

    @property
    def num_blobs_skipped(self):
        return self._num_blobs_skipped

    @property
    def blob_bytes(self):
        return self._blob_bytes

    @property
    def blob_bytes_skipped(self):
        return self._blob_bytes_skipped

    @property
    def plugin_names(self):
        return self._plugin_names

    def has_data(self):
        """Has any data been tracked by this instance.

        This counts the tensor and blob data that have been scanned
        but skipped.

        Returns:
          Whether this stats tracking object has tracked any data.
        """
        return self._num_scalars > 0 or self._num_tensors > 0 or self._num_blobs > 0

    def summarize(self):
        """Get a summary string for actually-uploaded and skipped data.

        Calling this property also marks the "last_summarized" timestamp, so that
        the has_new_data_since_last_summarize() will be able to report the correct value
        later.

        Returns:
          A tuple with two items:
          - A string summarizing all data uploaded so far.
          - If any data was skipped, a string for all skipped data. Else, `None`.
        """
        self._last_summarized_timestamp = time.time()
        string_pieces = []
        string_pieces.append('%d scalars' % self._num_scalars)
        uploaded_tensor_count = self._num_tensors - self._num_tensors_skipped
        uploaded_tensor_bytes = self._tensor_bytes - self._tensor_bytes_skipped
        string_pieces.append('0 tensors' if not uploaded_tensor_count else '%d tensors (%s)' % (uploaded_tensor_count, readable_bytes_string(uploaded_tensor_bytes)))
        uploaded_blob_count = self._num_blobs - self._num_blobs_skipped
        uploaded_blob_bytes = self._blob_bytes - self._blob_bytes_skipped
        string_pieces.append('0 binary objects' if not uploaded_blob_count else '%d binary objects (%s)' % (uploaded_blob_count, readable_bytes_string(uploaded_blob_bytes)))
        skipped_string = self._skipped_summary() if self._skipped_any() else None
        return (', '.join(string_pieces), skipped_string)

    def _skipped_any(self):
        """Whether any data was skipped."""
        return self._num_tensors_skipped or self._num_blobs_skipped

    def has_new_data_since_last_summarize(self):
        return self._last_data_added_timestamp > self._last_summarized_timestamp

    def _skipped_summary(self):
        """Get a summary string for skipped data."""
        string_pieces = []
        if self._num_tensors_skipped:
            string_pieces.append('%d tensors (%s)' % (self._num_tensors_skipped, readable_bytes_string(self._tensor_bytes_skipped)))
        if self._num_blobs_skipped:
            string_pieces.append('%d binary objects (%s)' % (self._num_blobs_skipped, readable_bytes_string(self._blob_bytes_skipped)))
        return ', '.join(string_pieces)

    def _refresh_last_data_added_timestamp(self):
        self._last_data_added_timestamp = time.time()