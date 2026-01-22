import abc
class BaseExperiment(metaclass=abc.ABCMeta):
    """Base class for experiment data access."""

    @abc.abstractmethod
    def get_scalars(self, runs_filter=None, tags_filter=None, pivot=False, include_wall_time=False):
        """Export scalar data as a pandas.DataFrame.

        Args:
          runs_filter: A regex filter for runs (e.g., r'run_[2-4]'). Operates in
            logical AND relation with `tags_filter`.
          tags_filter: A regex filter for tags (e.g., r'.*loss.*'). Operates in
            logical AND related with `runs_filter`.
          pivot: Whether to returned DataFrame will be pivoted (via pandas’
            `pivot_data()` method to a “wide” format wherein the tags of a
            given run and a given step are all collected in a single row.
            Setting `pivot` to `True` stipulates that the sets of step values
            are identical among all tags in every run of the experiment (after
            any run and tag filtering), so that the pivoting operation will not
            introduce missing values in the resultant DataFrame. Failing to meet
            this condition will cause `pivot=True` to raise a `ValueError`.
            If not provided, defaults to `False`.
          include_wall_time: Include wall_time (timestamps in nanoseconds since
            the epoch in float64) as a column in the returned DataFrame.
            If not provided, defaults to `False`.

        Returns:
          If `pivot` (default):
            A pivoted DataFrame with the indexing columns of
              - run
              - step
            And value columns that correspond to the tags.
            Duplicate entries for each run-step combination will be aggregated
            with `numpy.stack`. This format is more friendly to manipulation and
            plotting and hence io chosen as the default. When certain rows have
            missing values, a warning message will be displayed and advise the
            user to use the `pivot=False` if steps have different meanings in
            the experiment.
          If `not pivot`:
            A DataFrame with the following columns.
              - run: (non-null object)
              - tag: (non-null object)
              - steps: (non-null int64)
              - wall_time: (non-null object)
              - value: (non-null float32)
        """
        pass