from IPython.core.displayhook import DisplayHook
class SnapPyPromptDisplayHook(DisplayHook):
    """
    A DisplayHook used when displaying SnapPy's output prompts.  This
    subclass overrides one method in order to write the output prompt
    into the SnapPy console instead of sys.stdout.
    """

    def write_output_prompt(self):
        output = self.shell.output
        output.write(self.shell.separate_out)
        self.prompt_end_newline = True
        if self.do_full_cache:
            tokens = self.shell.prompts.out_prompt_tokens()
            prompt_txt = ''.join((s for t, s in tokens))
            if prompt_txt and (not prompt_txt.endswith('\n')):
                self.prompt_end_newline = False
            for token, text in tokens:
                output.write(text, style=(token[0],))