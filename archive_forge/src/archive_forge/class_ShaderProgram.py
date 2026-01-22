import logging
from OpenGL.GLES2 import *
from OpenGL._bytes import bytes,unicode,as_8_bit
class ShaderProgram(int):
    """Integer sub-class with context-manager operation"""

    def __enter__(self):
        """Start use of the program"""
        glUseProgram(self)

    def __exit__(self, typ, val, tb):
        """Stop use of the program"""
        glUseProgram(0)

    def check_validate(self):
        """Check that the program validates
        
        Validation has to occur *after* linking/loading
        
        raises RuntimeError on failures
        """
        glValidateProgram(self)
        validation = glGetProgramiv(self, GL_VALIDATE_STATUS)
        if validation == GL_FALSE:
            raise RuntimeError('Validation failure (%s): %s' % (validation, glGetProgramInfoLog(self)))
        return self

    def check_linked(self):
        """Check link status for this program
        
        raises RuntimeError on failures
        """
        link_status = glGetProgramiv(self, GL_LINK_STATUS)
        if link_status == GL_FALSE:
            raise RuntimeError('Link failure (%s): %s' % (link_status, glGetProgramInfoLog(self)))
        return self

    def retrieve(self):
        """Attempt to retrieve binary for this compiled shader
        
        Note that binaries for a program are *not* generally portable,
        they should be used solely for caching compiled programs for 
        local use; i.e. to reduce compilation overhead.
        
        returns (format,binaryData) for the shader program
        """
        from OpenGL.raw.GL._types import GLint, GLenum
        from OpenGL.arrays import GLbyteArray
        size = GLint()
        glGetProgramiv(self, get_program_binary.GL_PROGRAM_BINARY_LENGTH, size)
        result = GLbyteArray.zeros((size.value,))
        size2 = GLint()
        format = GLenum()
        get_program_binary.glGetProgramBinary(self, size.value, size2, format, result)
        return (format.value, result)

    def load(self, format, binary):
        """Attempt to load binary-format for a pre-compiled shader
        
        See notes in retrieve
        """
        get_program_binary.glProgramBinary(self, format, binary, len(binary))
        self.check_validate()
        self.check_linked()
        return self