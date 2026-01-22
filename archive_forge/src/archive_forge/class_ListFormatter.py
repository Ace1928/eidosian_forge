import abc
class ListFormatter(Formatter, metaclass=abc.ABCMeta):
    """Base class for formatters that know how to deal with multiple objects.
    """

    @abc.abstractmethod
    def emit_list(self, column_names, data, stdout, parsed_args):
        """Format and print the list from the iterable data source.

        Data values can be primitive types like ints and strings, or
        can be an instance of a :class:`FormattableColumn` for
        situations where the value is complex, and may need to be
        handled differently for human readable output vs. machine
        readable output.

        :param column_names: names of the columns
        :param data: iterable data source, one tuple per object
                     with values in order of column names
        :param stdout: output stream where data should be written
        :param parsed_args: argparse namespace from our local options

        """