import pyopencl as cl
import logging


class ShaderManager:
    def __init__(self, context):
        self.context = context

    def compile_shader(self, source, shader_type):
        if shader_type not in ["vertex", "fragment", "geometry"]:
            raise ValueError("Unsupported shader type provided.")
        # Create and build the program
        program = cl.Program(self.context, source).build()
        return program

    def load_shader_from_file(self, file_path, shader_type):
        with open(file_path, "r") as file:
            shader_source = file.read()
        return self.compile_shader(shader_source, shader_type)
