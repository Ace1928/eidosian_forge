import win32process

    This function mocks the generated pid aspect of the win32.CreateProcess
    function.
      - the true win32process.CreateProcess is called
      - return values are harvested in a tuple.
      - all return values from createProcess are passed back to the calling
        function except for the pid, the returned pid is hardcoded to 42
    