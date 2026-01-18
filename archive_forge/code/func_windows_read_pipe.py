import sys
def windows_read_pipe(fd, length):
    try:
        error, data = win32file.ReadFile(fd, length)
        return (error, data)
    except pywintypes.error as e:
        return (e.winerror, '')