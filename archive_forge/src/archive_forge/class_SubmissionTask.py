import copy
import logging
from s3transfer.utils import get_callbacks
class SubmissionTask(Task):
    """A base class for any submission task

    Submission tasks are the top-level task used to submit a series of tasks
    to execute a particular transfer.
    """

    def _main(self, transfer_future, **kwargs):
        """
        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future associated with the
            transfer request that tasks are being submitted for

        :param kwargs: Any additional kwargs that you may want to pass
            to the _submit() method
        """
        try:
            self._transfer_coordinator.set_status_to_queued()
            on_queued_callbacks = get_callbacks(transfer_future, 'queued')
            for on_queued_callback in on_queued_callbacks:
                on_queued_callback()
            self._transfer_coordinator.set_status_to_running()
            self._submit(transfer_future=transfer_future, **kwargs)
        except BaseException as e:
            self._log_and_set_exception(e)
            self._wait_for_all_submitted_futures_to_complete()
            self._transfer_coordinator.announce_done()

    def _submit(self, transfer_future, **kwargs):
        """The submition method to be implemented

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future associated with the
            transfer request that tasks are being submitted for

        :param kwargs: Any additional keyword arguments you want to be passed
            in
        """
        raise NotImplementedError('_submit() must be implemented')

    def _wait_for_all_submitted_futures_to_complete(self):
        submitted_futures = self._transfer_coordinator.associated_futures
        while submitted_futures:
            self._wait_until_all_complete(submitted_futures)
            possibly_more_submitted_futures = self._transfer_coordinator.associated_futures
            if submitted_futures == possibly_more_submitted_futures:
                break
            submitted_futures = possibly_more_submitted_futures