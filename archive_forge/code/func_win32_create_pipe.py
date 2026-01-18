import os
def win32_create_pipe():
    read_fd, write_fd = os.pipe()
    return (read_fd, write_fd)