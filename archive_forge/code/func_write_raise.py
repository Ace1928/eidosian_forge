def write_raise(self, stream, error_level=40, log_level=30):
    """Write report to `stream`

        Parameters
        ----------
        stream : file-like
           implementing ``write`` method
        error_level : int, optional
           level at which to raise error for problem detected in
           ``self``
        log_level : int, optional
           Such that if `log_level` is >= ``self.problem_level`` we
           write the report to `stream`, otherwise we write nothing.
        """
    if self.problem_level >= log_level:
        stream.write(f'Level {self.problem_level}: {self.message}\n')
    if self.problem_level and self.problem_level >= error_level:
        if self.error:
            raise self.error(self.problem_msg)